use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::super::MemoryService;
use crate::{
    error::NovaResult,
    memory::ops_remember::RememberInput,
    storage::{MemoryFilter, MemorySource, MemoryStatus, memory::MemoryRepository},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SummaryGroupBy {
    #[default]
    All,
    Tag,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SummaryOptions {
    pub enabled: bool,
    pub tag: Option<String>,
    pub min_group_size: usize,
    pub max_summary_chars: usize,
    pub max_members: usize,
    pub embed: bool,
    pub importance: f32,
}

impl Default for SummaryOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            tag: None,
            min_group_size: 3,
            max_summary_chars: 600,
            max_members: 50,
            embed: true,
            importance: 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEntry {
    pub summary_uuid: Uuid,
    pub member_uuids: Vec<Uuid>,
    pub original_chars: usize,
    pub summary_chars: usize,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummaryOutput {
    pub scanned: usize,
    pub entries: Vec<SummaryEntry>,
}

fn tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    for piece in text.split(|c: char| !c.is_alphanumeric()) {
        let lower = piece.to_lowercase();
        if lower.len() >= 2 {
            words.push(lower);
        }
    }
    words
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn extractive(texts: &[&str], max_chars: usize) -> String {
    let mut tf: HashMap<String, usize> = HashMap::new();
    for text in texts {
        for word in tokenize(text) {
            *tf.entry(word).or_insert(0) += 1;
        }
    }
    let mut candidates: Vec<(f32, usize, String)> = Vec::new();
    let mut order = 0usize;
    for text in texts {
        for sentence in split_sentences(text) {
            let words = tokenize(&sentence);
            if words.is_empty() {
                continue;
            }
            let sum: usize = words.iter().filter_map(|w| tf.get(w)).copied().sum();
            let score = sum as f32 / ((words.len() as f32) + 1.0).sqrt();
            candidates.push((score, order, sentence));
            order += 1;
        }
    }
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(&b.1))
    });
    let mut out = String::new();
    let limit = max_chars.max(1);
    for (_, _, sentence) in candidates {
        let sentence_chars = sentence.chars().count();
        let delta = if out.is_empty() { sentence_chars } else { sentence_chars + 1 };
        if out.chars().count() + delta > limit && !out.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&sentence);
    }
    out
}

pub async fn summarize_group(
    svc: &MemoryService,
    member_uuids: &[Uuid],
    opts: &SummaryOptions,
) -> NovaResult<SummaryEntry> {
    if !opts.enabled {
        return Err(crate::error::NovaError::validation("summary: feature disabled"));
    }
    let members = member_uuids.to_vec();
    let mut records = Vec::with_capacity(members.len());
    for uuid in &members {
        match svc.memory_repo.get_by_uuid(&svc.database, *uuid).await {
            Ok(r) => records.push(r),
            Err(_) => continue,
        }
    }
    if records.len() < opts.min_group_size.max(1) {
        return Err(crate::error::NovaError::validation(
            "summary: not enough members to summarize",
        ));
    }
    let texts: Vec<&str> = records.iter().map(|r| r.content.as_str()).collect();
    let summary = extractive(&texts, opts.max_summary_chars);
    if summary.is_empty() {
        return Err(crate::error::NovaError::validation("summary: empty extractive result"));
    }
    let member_uuids: Vec<Uuid> = records.iter().map(|r| r.uuid).collect();
    let original_chars: usize = records.iter().map(|r| r.content.chars().count()).sum();
    let summary_chars = summary.chars().count();
    let ratio = if original_chars > 0 { summary_chars as f32 / original_chars as f32 } else { 1.0 };
    let metas: Vec<String> = member_uuids.iter().map(|u| u.to_string()).collect();
    let metadata = serde_json::json!({
        "summary": true,
        "kind": "extractive",
        "members": metas,
        "compression_ratio": ratio,
    });
    let remember_out = svc
        .remember(RememberInput {
            content: &summary,
            source: MemorySource::System,
            importance: opts.importance,
            metadata: Some(&metadata),
            tags: &[],
            embed: opts.embed,
            extract_graph: false,
            ..Default::default()
        })
        .await?;
    Ok(SummaryEntry {
        summary_uuid: remember_out.uuid,
        member_uuids,
        original_chars,
        summary_chars,
        compression_ratio: ratio,
    })
}

pub async fn summarize_top(svc: &MemoryService, opts: SummaryOptions) -> NovaResult<SummaryOutput> {
    if !opts.enabled {
        return Ok(SummaryOutput::default());
    }
    let filter = MemoryFilter {
        status_in: Some(vec![MemoryStatus::Active]),
        tags_all: opts.tag.as_ref().map(|t| vec![t.clone()]),
        ..Default::default()
    };
    let limit = opts.max_members.clamp(2, 500);
    let records = svc.memory_repo.list(&svc.database, &filter, limit, 0).await?;
    let mut out = SummaryOutput {
        scanned: records.len(),
        entries: Vec::new(),
    };
    if records.len() < opts.min_group_size.max(1) {
        return Ok(out);
    }
    let uuids: Vec<Uuid> = records.iter().map(|r| r.uuid).collect();
    let entry = summarize_group(svc, &uuids, &opts).await?;
    out.entries.push(entry);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::StorageConfig,
        memory::ops_remember::{RememberInput, service_for_tests},
        storage::Database,
    };

    async fn temp_svc() -> crate::memory::MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-q2-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..Default::default()
        };
        let db = Database::open(cfg).await.unwrap();
        service_for_tests(db, 8, None)
    }

    #[test]
    fn extractive_produces_compressed_text() {
        let paragraphs = vec![
            "The quick brown fox jumps over the lazy dog near the river.",
            "The lazy dog sleeps under the warm sun all afternoon long.",
            "A quick fox chased the dog around the yard with great speed.",
        ];
        let summary = extractive(&paragraphs, 80);
        assert!(!summary.is_empty());
        assert!(summary.len() <= 80 + 40);
    }

    #[tokio::test]
    async fn summarize_group_creates_system_summary() {
        let svc = temp_svc().await;
        let mut uuids = Vec::new();
        for i in 0..5 {
            let out = svc
                .remember(RememberInput {
                    content: &format!("Project alpha sprint {} delivered on schedule.", i),
                    importance: 0.5,
                    ..Default::default()
                })
                .await
                .unwrap();
            uuids.push(out.uuid);
        }
        let opts = SummaryOptions {
            enabled: true,
            min_group_size: 3,
            max_summary_chars: 200,
            ..Default::default()
        };
        let entry = summarize_group(&svc, &uuids, &opts).await.unwrap();
        assert_eq!(entry.member_uuids.len(), 5);
        let summary = svc.get_memory(entry.summary_uuid).await.unwrap();
        assert_eq!(summary.source, MemorySource::System);
        assert_eq!(summary.metadata["summary"], serde_json::json!(true));
        assert!(summary.content.len() < 400);
    }

    #[tokio::test]
    async fn summarize_top_respects_tag_filter() {
        let svc = temp_svc().await;
        for i in 0..5 {
            svc.remember(RememberInput {
                content: &format!("backend memory number {}", i),
                tags: &["backend".to_string()],
                importance: 0.4,
                ..Default::default()
            })
            .await
            .unwrap();
        }
        svc.remember(RememberInput {
            content: "unique frontend memory",
            tags: &["frontend".to_string()],
            importance: 0.4,
            ..Default::default()
        })
        .await
        .unwrap();
        let opts = SummaryOptions {
            enabled: true,
            tag: Some("backend".to_string()),
            min_group_size: 3,
            max_summary_chars: 200,
            ..Default::default()
        };
        let out = summarize_top(&svc, opts).await.unwrap();
        assert_eq!(out.scanned, 5);
        assert_eq!(out.entries.len(), 1);
    }
}
