use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::extractor::{EntityCandidate, EntityExtractor, Extraction, RelationCandidate};
use crate::{
    embedding::retry::{RetryAction, RetryConfig, classify_http_status, with_retry},
    error::{NovaError, NovaResult},
};

pub const DEFAULT_EXTRACT_PROMPT: &str =
    "\
You are an information extraction engine working on memory notes.
Extract entities, relations and tags from the text below.
Return ONLY a single JSON object with exactly this shape:
{\"entities\":[{\"name\":\"...\",\"type\":\"...\",\"description\":\"...\"}],\"relations\":[{\"\
     source\":\"...\",\"target\":\"...\",\"predicate\":\"...\",\"confidence\":0.0}],\"tags\":[\"..\
     .\"]}
Rules:
1. Entities are named or significant concepts: people, organisations, places, products, events, \
     technical terms. Use a short lowercase type such as person, organisation, place, product, \
     event, concept, technology.
2. Relations connect two entity names that both appear in the text, with a short lowercase \
     predicate such as works_at, knows, located_in, part_of, uses, created. Set confidence in \
     [0,1] reflecting how strongly the text supports the relation.
3. Tags are 1 to 6 short lowercase keywords summarising the text.
4. Skip generic or trivial mentions. Do not invent facts that are not in the text.
5. If nothing meaningful is present, return {\"entities\":[],\"relations\":[],\"tags\":[]}.";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmExtractorConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(with = "crate::config::duration_seconds")]
    pub timeout: Duration,
    pub max_tokens: u32,
    pub temperature: f32,
    pub max_retries: u32,
}

impl Default for LlmExtractorConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            timeout: Duration::from_secs(30),
            max_tokens: 2048,
            temperature: 0.0,
            max_retries: 3,
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize, Default)]
struct LlmExtraction {
    #[serde(default)]
    entities: Vec<LlmEntity>,
    #[serde(default)]
    relations: Vec<LlmRelation>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LlmEntity {
    #[serde(default)]
    name: String,
    #[serde(default, rename = "type")]
    entity_type: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlmRelation {
    #[serde(default)]
    source: String,
    #[serde(default)]
    target: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    confidence: f32,
}

pub struct LLMEntityExtractor {
    client: reqwest::Client,
    config: LlmExtractorConfig,
    endpoint: String,
    prompt: String,
}

impl std::fmt::Debug for LLMEntityExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMEntityExtractor")
            .field("model", &self.config.model)
            .field("base_url", &self.config.base_url)
            .finish_non_exhaustive()
    }
}

impl LLMEntityExtractor {
    pub fn new(config: LlmExtractorConfig) -> NovaResult<Self> {
        Self::new_with_prompt(config, DEFAULT_EXTRACT_PROMPT.to_string())
    }

    pub fn new_with_prompt(config: LlmExtractorConfig, prompt: String) -> NovaResult<Self> {
        if config.base_url.trim().is_empty() {
            return Err(NovaError::config_msg("llm extractor: base_url must not be empty"));
        }
        if config.model.trim().is_empty() {
            return Err(NovaError::config_msg("llm extractor: model must not be empty"));
        }
        if !(0.0..=2.0).contains(&config.temperature) {
            return Err(NovaError::config_msg("llm extractor: temperature must be in [0, 2]"));
        }
        if config.max_tokens == 0 {
            return Err(NovaError::config_msg("llm extractor: max_tokens must be > 0"));
        }
        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = format!("{base}/chat/completions");
        let client = reqwest::Client::builder()
            .user_agent(concat!("yq-nova-agent/", env!("CARGO_PKG_VERSION")))
            .timeout(config.timeout)
            .build()
            .map_err(|e| NovaError::config_msg(format!("llm extractor: build client: {e}")))?;
        Ok(Self {
            client,
            config,
            endpoint,
            prompt,
        })
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    async fn chat(&self, text: &str) -> NovaResult<String> {
        let messages = vec![
            ChatMessage {
                role: "system",
                content: &self.prompt,
            },
            ChatMessage {
                role: "user",
                content: text,
            },
        ];
        let body = ChatRequest {
            model: &self.config.model,
            messages,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };
        let retry = RetryConfig {
            max_attempts: self.config.max_retries.max(1),
            base_sleep: Duration::from_millis(200),
            max_sleep: Duration::from_secs(2),
            per_attempt_timeout: self.config.timeout,
        };
        let response = with_retry(&retry, |_attempt| async {
            let mut request = self.client.post(self.endpoint.as_str()).json(&body);
            if !self.config.api_key.is_empty() {
                request = request.bearer_auth(&self.config.api_key);
            }
            let response = request.send().await.map_err(|e| {
                let action = if e.is_timeout() || e.is_connect() || e.is_request() {
                    RetryAction::Retry
                } else {
                    RetryAction::Fail
                };
                (action, anyhow::anyhow!("{e}"))
            })?;
            let status = response.status();
            if !status.is_success() {
                let raw = response.text().await.unwrap_or_default();
                let snippet = if raw.len() > 400 { &raw[..400] } else { raw.as_str() };
                return Err((
                    classify_http_status(status),
                    anyhow::anyhow!("HTTP {}: {}", status.as_u16(), snippet),
                ));
            }
            let parsed: ChatResponse = response
                .json()
                .await
                .map_err(|e| (RetryAction::Fail, anyhow::anyhow!("parse response: {e}")))?;
            Ok(parsed)
        })
        .await
        .map_err(|e| NovaError::graph_msg(format!("llm extraction request failed: {e}")))?;
        let content = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| NovaError::graph_msg("llm extraction: empty choices in response"))?;
        if content.trim().is_empty() {
            return Err(NovaError::graph_msg("llm extraction: empty content in response"));
        }
        Ok(content)
    }
}

fn extract_json_content(content: &str) -> Option<String> {
    let trimmed = content.trim();
    let stripped = trimmed.strip_prefix("```").unwrap_or(trimmed);
    let stripped = stripped.strip_suffix("```").unwrap_or(stripped);
    let start = stripped.find('{')?;
    let end = stripped.rfind('}')?;
    if end < start {
        return None;
    }
    Some(stripped[start..=end].to_string())
}

fn convert(raw: LlmExtraction) -> Extraction {
    let entities = raw
        .entities
        .into_iter()
        .filter_map(|entity| {
            let name = entity.name.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let entity_type = entity.entity_type.trim().to_string();
            let description =
                entity.description.map(|d| d.trim().to_string()).filter(|d| !d.is_empty());
            Some(EntityCandidate {
                name,
                entity_type: if entity_type.is_empty() {
                    "unknown".to_string()
                } else {
                    entity_type
                },
                description,
            })
        })
        .collect();
    let relations = raw
        .relations
        .into_iter()
        .filter_map(|relation| {
            let source_name = relation.source.trim().to_string();
            let target_name = relation.target.trim().to_string();
            let predicate = relation.predicate.trim().to_string();
            if source_name.is_empty() || target_name.is_empty() || predicate.is_empty() {
                return None;
            }
            if source_name == target_name {
                return None;
            }
            Some(RelationCandidate {
                source_name,
                target_name,
                predicate,
                confidence: relation.confidence.clamp(0.0, 1.0),
            })
        })
        .collect();
    let tags = raw
        .tags
        .into_iter()
        .map(|tag| tag.trim().to_string())
        .filter(|tag| !tag.is_empty())
        .collect();
    Extraction {
        entities,
        relations,
        tags,
    }
}

#[async_trait]
impl EntityExtractor for LLMEntityExtractor {
    async fn extract(&self, text: &str) -> NovaResult<Extraction> {
        if text.trim().is_empty() {
            return Ok(Extraction::default());
        }
        let content = self.chat(text).await?;
        let json = extract_json_content(&content).ok_or_else(|| {
            NovaError::graph_msg("llm extraction: no JSON object found in model output")
        })?;
        let raw: LlmExtraction = serde_json::from_str(&json)
            .map_err(|e| NovaError::graph_msg(format!("llm extraction: invalid JSON: {e}")))?;
        Ok(convert(raw))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    fn base_config() -> LlmExtractorConfig {
        LlmExtractorConfig {
            base_url: "http://127.0.0.1:1".to_string(),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
            timeout: Duration::from_secs(5),
            max_tokens: 128,
            temperature: 0.0,
            max_retries: 1,
        }
    }

    async fn spawn_mock_server(body: String, status: u16) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0u8; 8192];
            let mut read = 0usize;
            loop {
                let n = socket.read(&mut buffer[read..]).await.unwrap();
                if n == 0 {
                    break;
                }
                read += n;
                if buffer[..read].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let reason = if status == 200 { "OK" } else { "Error" };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: \
                 {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        });
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn extracts_entities_relations_and_tags_from_fenced_output() {
        let content = "```json\n{\"entities\":[{\"name\":\"Alice\",\"type\":\"person\"},{\"name\":\
                       \"Acme Corp\",\"type\":\"organisation\",\"description\":\"a \
                       company\"}],\"relations\":[{\"source\":\"Alice\",\"target\":\"Acme \
                       Corp\",\"predicate\":\"works_at\",\"confidence\":0.95}],\"tags\":[\"work\",\
                       \"meeting\"]}\n```";
        let body =
            serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string();
        let (base, server) = spawn_mock_server(body, 200).await;
        let config = LlmExtractorConfig {
            base_url: base,
            ..base_config()
        };
        let extractor = LLMEntityExtractor::new(config).unwrap();
        let out = extractor.extract("Alice works at Acme Corp.").await.unwrap();
        assert_eq!(out.entities.len(), 2);
        assert!(out.entities.iter().any(|e| e.name == "Alice" && e.entity_type == "person"));
        assert!(
            out.entities.iter().any(|e| e.name == "Acme Corp" && e.entity_type == "organisation")
        );
        assert_eq!(out.relations.len(), 1);
        let rel = &out.relations[0];
        assert_eq!(rel.source_name, "Alice");
        assert_eq!(rel.target_name, "Acme Corp");
        assert_eq!(rel.predicate, "works_at");
        assert!((rel.confidence - 0.95).abs() < 1e-6);
        assert_eq!(out.tags, vec!["work".to_string(), "meeting".to_string()]);
        server.abort();
    }

    #[tokio::test]
    async fn plain_json_output_is_parsed() {
        let content = "{\"entities\":[{\"name\":\"Rust\",\"type\":\"technology\"}],\"relations\":\
                       [],\"tags\":[\"lang\"]}";
        let body =
            serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string();
        let (base, server) = spawn_mock_server(body, 200).await;
        let config = LlmExtractorConfig {
            base_url: base,
            ..base_config()
        };
        let extractor = LLMEntityExtractor::new(config).unwrap();
        let out = extractor.extract("Rust is a language").await.unwrap();
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].name, "Rust");
        assert_eq!(out.entities[0].entity_type, "technology");
        assert_eq!(out.tags, vec!["lang".to_string()]);
        server.abort();
    }

    #[tokio::test]
    async fn empty_extraction_is_accepted() {
        let content = "{\"entities\":[],\"relations\":[],\"tags\":[]}";
        let body =
            serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string();
        let (base, server) = spawn_mock_server(body, 200).await;
        let config = LlmExtractorConfig {
            base_url: base,
            ..base_config()
        };
        let extractor = LLMEntityExtractor::new(config).unwrap();
        let out = extractor.extract("nothing here").await.unwrap();
        assert!(out.entities.is_empty());
        assert!(out.relations.is_empty());
        assert!(out.tags.is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn server_error_fails_fast_with_single_retry() {
        let (base, server) = spawn_mock_server("boom".to_string(), 500).await;
        let config = LlmExtractorConfig {
            base_url: base,
            ..base_config()
        };
        let extractor = LLMEntityExtractor::new(config).unwrap();
        assert!(extractor.extract("x").await.is_err());
        server.abort();
    }

    #[tokio::test]
    async fn non_json_output_is_rejected() {
        let body =
            serde_json::json!({ "choices": [{ "message": { "content": "sorry, no json" } }] })
                .to_string();
        let (base, server) = spawn_mock_server(body, 200).await;
        let config = LlmExtractorConfig {
            base_url: base,
            ..base_config()
        };
        let extractor = LLMEntityExtractor::new(config).unwrap();
        assert!(extractor.extract("x").await.is_err());
        server.abort();
    }

    #[test]
    fn new_validates_config() {
        let bad_model = LlmExtractorConfig {
            model: "  ".to_string(),
            ..base_config()
        };
        assert!(LLMEntityExtractor::new(bad_model).is_err());
        let bad_temp = LlmExtractorConfig {
            temperature: 3.0,
            ..base_config()
        };
        assert!(LLMEntityExtractor::new(bad_temp).is_err());
        let zero_tokens = LlmExtractorConfig {
            max_tokens: 0,
            ..base_config()
        };
        assert!(LLMEntityExtractor::new(zero_tokens).is_err());
    }

    #[test]
    fn json_extraction_handles_fences_and_prefix_text() {
        let raw = "Here you go: ```json\n{\"entities\":[]}\n```";
        assert_eq!(extract_json_content(raw).as_deref(), Some("{\"entities\":[]}"));
        assert_eq!(extract_json_content("no braces").as_deref(), None);
    }

    #[test]
    fn convert_skips_invalid_candidates() {
        let raw = LlmExtraction {
            entities: vec![
                LlmEntity {
                    name: "  ".to_string(),
                    entity_type: "x".to_string(),
                    description: None,
                },
                LlmEntity {
                    name: "  Valid  ".to_string(),
                    entity_type: "  ".to_string(),
                    description: Some("  ".to_string()),
                },
            ],
            relations: vec![
                LlmRelation {
                    source: "A".to_string(),
                    target: "A".to_string(),
                    predicate: "self".to_string(),
                    confidence: 1.0,
                },
                LlmRelation {
                    source: "A".to_string(),
                    target: "B".to_string(),
                    predicate: "knows".to_string(),
                    confidence: 1.7,
                },
            ],
            tags: vec!["  ".to_string(), "ok".to_string()],
        };
        let out = convert(raw);
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].name, "Valid");
        assert_eq!(out.entities[0].entity_type, "unknown");
        assert_eq!(out.relations.len(), 1);
        assert_eq!(out.relations[0].confidence, 1.0);
        assert_eq!(out.tags, vec!["ok".to_string()]);
    }
}
