//! Graph / knowledge-extractor traits + MVP regex-based extractor.
//!
//! Proper entity extraction is expensive and model-specific; the MVP ships a
//! deliberately dumb extractor that only catches:
//!   * `[[WikiLinks]]`-style double-bracketed names (used in notes / memory
//!     systems that already mark entities explicitly)
//!   * Capitalised multi-word sequences ("Alice Smith", "Kubernetes") when
//!     they appear outside leading-word positions.
//!
//! Both forms are intentionally conservative — the point is not to be smart,
//! it's to exercise the *graph plumbing* (entities, relations, traversal)
//! so users can plug in a spaCy / LLM extractor later without touching the
//! rest of the codebase. The trait is simple enough that this replacement
//! is a one-file change.

use std::collections::BTreeSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::NovaResult;

/// A candidate entity produced by the extractor. Extractor-only type; actual
/// persistence happens through `EntityRepository::upsert` which may merge names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityCandidate {
    /// Canonical entity name. Trimmed, no brackets.
    pub name: String,
    /// Coarse type guess: `"person"` / `"org"` / `"unknown"` / etc. The MVP
    /// regex extractor only ever emits `"unknown"` + WikiLinks get a type of
    /// `"wiki"`. Downstream callers normalise freely.
    pub entity_type: String,
    /// Optional short description the extractor picked up. MVP always empty.
    pub description: Option<String>,
}

/// A candidate directed relation between two entities. Again MVP emits only the
/// explicit `"mentions"` predicate for any co-occurring capitalised pairs
/// within the same memory. LLM / NER extractors can enrich later.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationCandidate {
    pub source_name: String,
    pub target_name: String,
    pub predicate: String,
    /// Estimated confidence ∈ [0, 1]. MVP uses fixed values.
    pub confidence: f32,
}

/// Result of running an extractor over a single text.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Extraction {
    pub entities: Vec<EntityCandidate>,
    pub relations: Vec<RelationCandidate>,
    /// Optional short tag phrases pulled from the text (e.g. "#tag").
    /// Attached to the memory as tags, not persisted to the graph.
    pub tags: Vec<String>,
}

#[async_trait]
pub trait EntityExtractor: Send + Sync + std::fmt::Debug {
    async fn extract(&self, text: &str) -> NovaResult<Extraction>;
}

// ---------------------------------------------------------------------------
// MVP: RegexWikiExtractor
// ---------------------------------------------------------------------------

/// Conservative rule-based extractor. Zero model, zero deps. Enough to exercise
/// the graph layer end-to-end until a real NLP extractor is wired in.
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
    // [[Name]] or [[Name|Type]]
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
    // very small hand-rolled scanner: #tagname where tagname is [A-Za-z0-9_-]+
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

/// Extract capitalised proper noun candidates (ASCII-only, runs of [A-Z][a-zA-Z]+ separated
/// by spaces, at least one interior capital letter). Position-ignorant for the MVP
/// because the point is exercising the graph plumbing.
fn extract_capitalised(text: &str, out: &mut Extraction) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        // Find start of candidate: an ASCII uppercase letter, preceded by a
        // non-alphanumeric (so we don't match mid-word).
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
        // Walk the run: tokens separated by exactly one space between
        // capitalised words, e.g. "Alice M. Smith".
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
                // Next char after a single space must be ASCII uppercase.
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
        // Right boundary: a token can't end with a space.
        let slice_end = if chars[end - 1] == ' ' { end - 1 } else { end };
        if word_count >= 1 {
            let name: String = chars[i..slice_end].iter().collect();
            let name = name.trim().to_string();
            // Skip stop words and tiny tokens that are probably just sentence starters.
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
    // Pair every distinct entity with every other in this memory with a
    // "mentions" edge. Low confidence (0.3 by default) because co-occurrence is
    // weak evidence; stronger extractors can emit higher.
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

// A completely inert extractor — every call returns empty. Used as default.
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
        // At least one mentions edge between Alice Smith / Bob Jones.
        assert!(r.relations.iter().any(|rel| rel.predicate == "mentions"));
    }
}
