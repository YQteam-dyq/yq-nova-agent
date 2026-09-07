
use std::collections::BTreeSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::NovaResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityCandidate {

    pub name: String,

    pub entity_type: String,

    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationCandidate {
    pub source_name: String,
    pub target_name: String,
    pub predicate: String,

    pub confidence: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Extraction {
    pub entities: Vec<EntityCandidate>,
    pub relations: Vec<RelationCandidate>,

    pub tags: Vec<String>,
}

#[async_trait]
pub trait EntityExtractor: Send + Sync + std::fmt::Debug {
    async fn extract(&self, text: &str) -> NovaResult<Extraction>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RegexWikiExtractor;

impl RegexWikiExtractor {
    pub fn new() -> Self {
        Self
    }
}

fn dedupe_keep_order_by_name<T: Clone>(items: Vec<T>, key: impl Fn(&T) -> &str) -> Vec<T> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        if seen.insert(key(&it).to_string()) {
            out.push(it);
        }
    }
    out
}

fn extract_wikilinks(text: &str, out: &mut Extraction) {

    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            let start = i + 2;
            let mut end = None;
            let mut j = start;
            while j + 1 < bytes.len() {
                if bytes[j] == b']' && bytes[j + 1] == b']' {
                    end = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(end) = end {
                let inner = &text[start..end];
                let (name, etype) = match inner.split_once('|') {
                    Some((n, t)) => (n.trim().to_string(), format!("wiki:{}", t.trim())),
                    None => (inner.trim().to_string(), "wiki".to_string()),
                };
                if !name.is_empty() {
                    out.entities.push(EntityCandidate {
                        name,
                        entity_type: etype,
                        description: None,
                    });
                }
                i = end + 2;
                continue;
            }
        }
        i += 1;
    }
}

fn extract_hashtags(text: &str, out: &mut Extraction) {

    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#'
            && (i == 0 || !is_tag_char(bytes[i - 1]))
            && i + 1 < bytes.len()
            && is_tag_start(bytes[i + 1])
        {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && is_tag_char(bytes[j]) {
                j += 1;
            }
            let tag = text[start..j].to_string();
            if !tag.is_empty() {
                out.tags.push(tag);
            }
            i = j;
            continue;
        }
        i += 1;
    }

    fn is_tag_start(b: u8) -> bool {
        b.is_ascii_alphabetic()
    }
    fn is_tag_char(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
    }
}

fn extract_capitalised(text: &str, out: &mut Extraction) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {

        let c0 = chars[i];
        if !c0.is_ascii_uppercase() {
            i += 1;
            continue;
        }
        let ok_boundary_left = i == 0 || {
            let prev = chars[i - 1];
            !prev.is_ascii_alphanumeric()
        };
        if !ok_boundary_left {
            i += 1;
            continue;
        }

        let mut end = i + 1;
        let mut last_was_space = false;
        let mut word_count = 1u32;
        while end < n {
            let c = chars[end];
            if c == ' ' && !last_was_space {
                last_was_space = true;
                end += 1;
                continue;
            }
            if last_was_space {

                if c.is_ascii_uppercase() {
                    last_was_space = false;
                    word_count += 1;
                    end += 1;
                    continue;
                } else {
                    break;
                }
            }
            if c.is_ascii_alphabetic() || c == '.' || c == '\'' {
                end += 1;
                continue;
            }
            break;
        }

        let slice_end = if chars[end - 1] == ' ' { end - 1 } else { end };
        if word_count >= 1 {
            let name: String = chars[i..slice_end].iter().collect();
            let name = name.trim().to_string();

            if name.len() >= 2 && !is_common_sentence_opener(&name) {
                out.entities.push(EntityCandidate {
                    name,
                    entity_type: "unknown".to_string(),
                    description: None,
                });
            }
        }
        i = (end + 1).min(n);
    }

    fn is_common_sentence_opener(name: &str) -> bool {
        matches!(
            name,
            "The"
                | "A"
                | "An"
                | "This"
                | "That"
                | "These"
                | "Those"
                | "I"
                | "We"
                | "You"
                | "He"
                | "She"
                | "It"
                | "They"
                | "And"
                | "But"
                | "So"
                | "If"
                | "When"
                | "Hello"
                | "Hi"
                | "Hey"
                | "Thanks"
                | "Thank"
                | "Please"
        )
    }
}

fn emit_cooccurrence_relations(out: &mut Extraction, confidence: f32) {

    let entities: Vec<String> = out.entities.iter().map(|e| e.name.clone()).collect();
    for (i, a) in entities.iter().enumerate() {
        for b in entities.iter().skip(i + 1) {
            if a != b {
                out.relations.push(RelationCandidate {
                    source_name: a.clone(),
                    target_name: b.clone(),
                    predicate: "mentions".to_string(),
                    confidence,
                });
            }
        }
    }
}

#[async_trait]
impl EntityExtractor for RegexWikiExtractor {
    async fn extract(&self, text: &str) -> NovaResult<Extraction> {
        let mut out = Extraction::default();
        extract_wikilinks(text, &mut out);
        extract_hashtags(text, &mut out);
        extract_capitalised(text, &mut out);
        out.entities = dedupe_keep_order_by_name(out.entities, |e| e.name.as_str());
        out.tags = dedupe_keep_order_by_name(out.tags, |t| t.as_str());
        emit_cooccurrence_relations(&mut out, 0.3_f32);
        Ok(out)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopExtractor;

#[async_trait]
impl EntityExtractor for NoopExtractor {
    async fn extract(&self, _text: &str) -> NovaResult<Extraction> {
        Ok(Extraction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_always_empty() {
        let e = NoopExtractor;
        let r = e.extract("anything [[A]] #tag").await.unwrap();
        assert!(r.entities.is_empty());
        assert!(r.relations.is_empty());
        assert!(r.tags.is_empty());
    }

    #[tokio::test]
    async fn wikilink_extracts_name_and_wiki_type() {
        let e = RegexWikiExtractor::new();
        let r = e.extract("see [[Rust|lang]] and [[Trae]] for details").await.unwrap();
        assert!(r.entities.iter().any(|x| x.name == "Rust" && x.entity_type == "wiki:lang"));
        assert!(r.entities.iter().any(|x| x.name == "Trae" && x.entity_type == "wiki"));
    }

    #[tokio::test]
    async fn hashtags_are_pulled() {
        let e = RegexWikiExtractor::new();
        let r = e.extract("deploy #prod and #k8s-1.30, skip no#match").await.unwrap();
        assert_eq!(r.tags, vec!["prod".to_string(), "k8s-1.30".to_string()]);
    }

    #[tokio::test]
    async fn capitalised_entities_and_cooccurrence() {
        let e = RegexWikiExtractor::new();
        let r = e.extract("Alice Smith met Bob Jones at Acme Corp last Tuesday").await.unwrap();
        let names: Vec<&str> = r.entities.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"Alice Smith"), "names={names:?}");
        assert!(names.contains(&"Bob Jones"), "names={names:?}");
        assert!(names.contains(&"Acme Corp"), "names={names:?}");

        assert!(r.relations.iter().any(|rel| rel.predicate == "mentions"));
    }
}
