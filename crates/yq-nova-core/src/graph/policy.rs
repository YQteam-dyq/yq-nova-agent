use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::extractor::{EntityExtractor, Extraction};
use crate::error::NovaResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionFrequency {
    #[default]
    PerWrite,
    Interval {
        every_secs: u64,
    },
    Never,
}

impl ExtractionFrequency {
    pub fn should_extract(&self, last_extracted_at: Option<chrono::DateTime<chrono::Utc>>) -> bool {
        match self {
            ExtractionFrequency::PerWrite => true,
            ExtractionFrequency::Never => false,
            ExtractionFrequency::Interval {
                every_secs,
            } => match last_extracted_at {
                None => true,
                Some(last) => {
                    let elapsed = chrono::Utc::now().signed_duration_since(last);
                    elapsed.num_seconds() >= *every_secs as i64
                },
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractionPolicy {
    pub frequency: ExtractionFrequency,
    pub min_confidence: f32,
    pub entity_type_whitelist: Vec<String>,
    pub relation_type_whitelist: Vec<String>,
}

impl Default for ExtractionPolicy {
    fn default() -> Self {
        Self {
            frequency: ExtractionFrequency::PerWrite,
            min_confidence: 0.0,
            entity_type_whitelist: Vec::new(),
            relation_type_whitelist: Vec::new(),
        }
    }
}

impl ExtractionPolicy {
    pub fn is_entity_type_allowed(&self, entity_type: &str) -> bool {
        self.entity_type_whitelist.is_empty()
            || self
                .entity_type_whitelist
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(entity_type.trim()))
    }

    pub fn is_relation_allowed(&self, predicate: &str, confidence: f32) -> bool {
        if confidence < self.min_confidence {
            return false;
        }
        self.relation_type_whitelist.is_empty()
            || self
                .relation_type_whitelist
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(predicate.trim()))
    }

    pub fn apply(&self, extraction: Extraction) -> Extraction {
        let mut out = extraction;
        if !self.entity_type_whitelist.is_empty() {
            out.entities.retain(|entity| self.is_entity_type_allowed(&entity.entity_type));
        }
        out.relations
            .retain(|relation| self.is_relation_allowed(&relation.predicate, relation.confidence));
        out
    }
}

#[derive(Clone)]
pub struct FilteringExtractor {
    inner: std::sync::Arc<dyn EntityExtractor>,
    policy: ExtractionPolicy,
    last_extracted_at: Option<DateTime<Utc>>,
}

impl std::fmt::Debug for FilteringExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteringExtractor").field("policy", &self.policy).finish_non_exhaustive()
    }
}

impl FilteringExtractor {
    pub fn new(inner: std::sync::Arc<dyn EntityExtractor>, policy: ExtractionPolicy) -> Self {
        Self {
            inner,
            policy,
            last_extracted_at: None,
        }
    }

    pub fn with_last_extracted_at(mut self, last_extracted_at: Option<DateTime<Utc>>) -> Self {
        self.last_extracted_at = last_extracted_at;
        self
    }

    pub fn policy(&self) -> &ExtractionPolicy {
        &self.policy
    }
}

#[async_trait]
impl EntityExtractor for FilteringExtractor {
    async fn extract(&self, text: &str) -> NovaResult<Extraction> {
        if !self.policy.frequency.should_extract(self.last_extracted_at) {
            return Ok(Extraction::default());
        }
        let extraction = self.inner.extract(text).await?;
        Ok(self.policy.apply(extraction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extractor::{EntityCandidate, RegexWikiExtractor, RelationCandidate};

    fn sample_extraction() -> Extraction {
        Extraction {
            entities: vec![
                EntityCandidate {
                    name: "Alice".to_string(),
                    entity_type: "person".to_string(),
                    description: None,
                },
                EntityCandidate {
                    name: "Acme Corp".to_string(),
                    entity_type: "organisation".to_string(),
                    description: None,
                },
            ],
            relations: vec![
                RelationCandidate {
                    source_name: "Alice".to_string(),
                    target_name: "Acme Corp".to_string(),
                    predicate: "works_at".to_string(),
                    confidence: 0.9,
                },
                RelationCandidate {
                    source_name: "Alice".to_string(),
                    target_name: "Acme Corp".to_string(),
                    predicate: "visits".to_string(),
                    confidence: 0.4,
                },
            ],
            tags: vec!["work".to_string()],
        }
    }

    #[test]
    fn entity_whitelist_filters_by_type_case_insensitively() {
        let policy = ExtractionPolicy {
            entity_type_whitelist: vec!["Person".to_string()],
            ..Default::default()
        };
        let out = policy.apply(sample_extraction());
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].name, "Alice");
        assert_eq!(out.relations.len(), 2);
    }

    #[test]
    fn relation_whitelist_and_confidence_threshold_filter_relations() {
        let policy = ExtractionPolicy {
            min_confidence: 0.6,
            relation_type_whitelist: vec!["works_at".to_string()],
            ..Default::default()
        };
        let out = policy.apply(sample_extraction());
        assert_eq!(out.entities.len(), 2);
        assert_eq!(out.relations.len(), 1);
        assert_eq!(out.relations[0].predicate, "works_at");
    }

    #[test]
    fn empty_whitelists_keep_everything_above_floor() {
        let policy = ExtractionPolicy {
            min_confidence: 0.5,
            ..Default::default()
        };
        let out = policy.apply(sample_extraction());
        assert_eq!(out.entities.len(), 2);
        assert_eq!(out.relations.len(), 1);
        assert_eq!(out.relations[0].predicate, "works_at");
    }

    #[test]
    fn frequency_gates_extraction() {
        assert!(ExtractionFrequency::PerWrite.should_extract(None));
        assert!(ExtractionFrequency::PerWrite.should_extract(Some(chrono::Utc::now())));
        assert!(!ExtractionFrequency::Never.should_extract(None));
        let interval = ExtractionFrequency::Interval {
            every_secs: 60,
        };
        assert!(interval.should_extract(None));
        let recent = chrono::Utc::now() - chrono::Duration::seconds(10);
        assert!(!interval.should_extract(Some(recent)));
        let old = chrono::Utc::now() - chrono::Duration::seconds(120);
        assert!(interval.should_extract(Some(old)));
    }

    #[tokio::test]
    async fn filtering_extractor_wraps_inner_extractor() {
        let inner = std::sync::Arc::new(RegexWikiExtractor::new());
        let policy = ExtractionPolicy {
            entity_type_whitelist: vec!["wiki:lang".to_string()],
            ..Default::default()
        };
        let extractor = FilteringExtractor::new(inner, policy);
        let out = extractor.extract("see [[Rust|lang]] and [[Trae]] for details").await.unwrap();
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].name, "Rust");
    }
}
