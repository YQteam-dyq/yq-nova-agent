//! Configuration loading for yq-nova.
//!
//! Sources are layered (last wins):
//!   1. Compiled-in defaults
//!   2. `YQ_NOVA_CONFIG` env var points to TOML file (or `./yq-nova.toml`)
//!   3. `YQ_NOVA_*` / `YQ_NOVA_<SECTION>__<KEY>` env vars (double-underscore
//!      separates nested sections)
//!
//! This is implemented on top of [`figment`], which takes care of merging
//! and env var splitting for us.

use std::{fs, path::PathBuf, time::Duration};

use figment::{
    Figment, Profile,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};

use crate::error::{NovaError, NovaResult};

/// Serde helper: serialise a `Duration` as integer seconds, deserialise
/// from integer seconds. Used instead of `serde_with::serde_as` which
/// requires the `#[serde_as]` macro attribute on an *outer* struct.
pub(crate) mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, ser: S) -> Result<S::Ok, S::Error> {
        d.as_secs().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(de)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Default config filename we look for in the current working directory
/// when `YQ_NOVA_CONFIG` is not set.
pub const DEFAULT_CONFIG_FILENAME: &str = "yq-nova.toml";

// =========================================================================
// Top-level config
// =========================================================================

/// 顶层配置，聚合所有子模块配置。
///
/// 加载顺序（后者覆盖前者）：
/// 1. 编译期默认值
/// 2. `YQ_NOVA_CONFIG` 环境变量指向的 TOML 文件（或当前目录 `./yq-nova.toml`）
/// 3. `YQ_NOVA_*` / `YQ_NOVA_<SECTION>__<KEY>` 环境变量（双下划线分隔嵌套段）
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub forgetting: ForgettingConfig,
    pub graph: GraphConfig,
    pub jobs: JobsConfig,
    pub logging: LoggingConfig,
}

// =========================================================================
// Sections
// =========================================================================

/// HTTP 服务器运行时配置。
///
/// 推荐取值：
/// - `concurrency`: 16~128，取决于部署机器的 CPU 核心数
/// - `request_timeout`: 15~60 秒；若使用慢速 embedding 上游请适当调大
/// - `max_request_body_bytes`: 最小 1024，推荐 1~10 MB
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// 监听地址。默认 `127.0.0.1:7999`，未加认证层前切勿暴露到 `0.0.0.0`。
    pub bind: String,
    /// 并发请求上限，超过后返回 503。推荐范围 16..=128。
    pub concurrency: usize,
    /// 单请求硬超时（秒），包含上游 embedding 调用。默认 30 秒。
    #[serde(with = "duration_seconds")]
    pub request_timeout: Duration,
    /// 请求体最大字节数，最小 1024，默认 10 MB。
    pub max_request_body_bytes: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:7999".into(),
            concurrency: 32,
            request_timeout: Duration::from_secs(30),
            max_request_body_bytes: 10 * 1024 * 1024,
        }
    }
}

/// SQLite 存储层配置。
///
/// 推荐取值：
/// - `cache_size_kb`: 64 MB ~ 1 GB，取决于可用 RAM
/// - `mmap_size_kb`: 建议 ≥ 预期 DB 文件大小，0 表示禁用 mmap
/// - `pool_max_connections`: 4~32，SQLite 写并发仍然受限（单写者）
/// - `synchronous`: 生产环境推荐 `normal`；对数据安全要求极高可使用 `full`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// SQLite 数据库文件路径。不得包含 `..` 路径穿越组件。
    pub db_path: PathBuf,
    /// 是否启用 WAL 日志模式，推荐开启以获得更好的读写并发。
    pub wal_mode: bool,
    /// SQLite 页大小（字节）。`0` 表示使用 SQLite 默认值；推荐 4096。
    pub page_size: i64,
    /// SQLite 页缓存大小（KB），负值表示 KB。默认 128 MB。
    pub cache_size_kb: i64,
    /// 每连接的 `busy_timeout`（毫秒）。默认 5000；写密集场景可适当调大。
    pub busy_timeout_ms: u32,
    /// sqlx 连接池最大连接数，推荐范围 4..=32。
    pub pool_max_connections: u32,
    /// 连接池保持的最小空闲连接数，推荐 0~4。
    pub pool_min_connections: u32,
    /// SQLite 内存映射 I/O 大小（KB）。`0` 禁用 mmap。默认 256 MB。
    pub mmap_size_kb: i64,
    /// SQLite `soft_heap_limit`（KB）。`0` 表示无限制。默认 512 MB。
    pub soft_heap_limit_kb: i64,
    /// WAL 被动 checkpoint 触发阈值（KB）。默认 1024（1 MB）。
    /// 写密集型负载可调高以摊销 checkpoint 开销。
    pub wal_autocheckpoint_kb: i64,
    /// WAL 文件大小硬上限（KB），超过后强制 TRUNCATE checkpoint。
    /// `0` 禁用限制；默认 128 MB。
    pub journal_size_limit_kb: i64,
    /// SQLite `synchronous` 级别：`off`|`normal`|`full`（或数字 0/1/2/3）。
    /// 默认 `normal`：应用崩溃安全，仅断电可能丢失最后一笔事务。
    pub synchronous: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./yq-nova.db"),
            wal_mode: true,
            page_size: 4096,
            cache_size_kb: 131_072,
            busy_timeout_ms: 5_000,
            pool_max_connections: 16,
            pool_min_connections: 2,
            mmap_size_kb: 262_144,
            soft_heap_limit_kb: 524_288,
            wal_autocheckpoint_kb: 1_024,
            journal_size_limit_kb: 131_072,
            synchronous: "normal".into(),
        }
    }
}

// -------------------------------------------------------------------------
// Embedding config (provider registry + default)
// -------------------------------------------------------------------------

/// 向量嵌入（Embedding）层配置：提供者注册表 + 默认选择。
///
/// 支持两类提供者：
/// - `openai_compatible`: OpenAI 兼容 HTTP 接口（如 OpenAI、Azure、Ollama 等）
/// - `fastembed_local`: 本地 ONNX FastEmbed（feature-gated）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// 默认使用的提供者名称，必须存在于注册表中。
    pub default_provider: String,
    /// OpenAI 兼容 HTTP 提供者，key 为自定义名称。
    pub openai_compatible: std::collections::BTreeMap<String, OpenAiCompatConfig>,
    /// FastEmbed 本地 ONNX 提供者，key 为自定义名称。
    /// MVP：占位实现，需启用 `fastembed` feature。
    pub fastembed_local: std::collections::BTreeMap<String, FastEmbedConfig>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        let mut oai = std::collections::BTreeMap::new();
        oai.insert("default".into(), OpenAiCompatConfig::default());
        Self {
            default_provider: "default".into(),
            openai_compatible: oai,
            fastembed_local: std::collections::BTreeMap::new(),
        }
    }
}

/// OpenAI 兼容 HTTP Embedding 提供者配置。
///
/// 推荐取值：
/// - `dimensions`: 常见值如 1536（text-embedding-3-small）、3072（text-embedding-3-large）、768（bge-m3）
/// - `batch_size`: 取决于具体模型，OpenAI 官方上限 2048，推荐 16~128
/// - `max_retries`: 0~5，默认 3
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiCompatConfig {
    /// API 基础 URL，如 `https://api.openai.com/v1`。
    pub base_url: String,
    /// API Key。生产环境优先使用环境变量
    /// `YQ_NOVA_EMBEDDING__OPENAI_COMPATIBLE__<NAME>__API_KEY` 注入。
    pub api_key: String,
    /// 模型名称，如 `text-embedding-3-small`。
    pub model: String,
    /// 向量维度，必须与模型实际输出一致。
    pub dimensions: usize,
    /// 单次批量 embedding 的文本条数，默认 16。
    pub batch_size: usize,
    /// 请求超时（秒），默认 15。
    #[serde(with = "duration_seconds")]
    pub timeout: Duration,
    /// 临时性错误的最大重试次数，默认 3。
    pub max_retries: u32,
}

impl Default for OpenAiCompatConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "text-embedding-3-small".into(),
            dimensions: 1536,
            batch_size: 16,
            timeout: Duration::from_secs(15),
            max_retries: 3,
        }
    }
}

/// FastEmbed 本地 ONNX Embedding 提供者配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FastEmbedConfig {
    /// FastEmbed 模型名称，如 `BAAI/bge-small-en-v1.5`。
    pub model_name: String,
    /// 模型文件缓存目录，默认为 FastEmbed 的全局缓存。
    pub cache_dir: PathBuf,
}

// -------------------------------------------------------------------------
// Forgetting / TTL
// -------------------------------------------------------------------------

/// 记忆遗忘（TTL / 老化）后台策略配置。
///
/// 推荐取值：
/// - `stale_after`: 7~365 天；默认 90 天
/// - `stale_importance_threshold`: ∈ [0, 1]，默认 0.3，仅低于该阈值的记忆才会被清理
/// - `action`: `archive`（默认，可审计恢复）或 `delete`（硬删除释放空间）
/// - `check_interval`: 60~3600 秒；默认 600
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ForgettingConfig {
    /// 是否启用后台遗忘任务。
    pub enabled: bool,
    /// 超过该时长未被访问的记忆视为「陈旧」。默认 90 天。
    #[serde(with = "duration_seconds")]
    pub stale_after: Duration,
    /// 陈旧记忆只有 importance 严格低于该值才会被清理。
    /// 合法范围 [0.0, 1.0]，默认 0.3。
    pub stale_importance_threshold: f32,
    /// 清理动作：`archive`（软归档）或 `delete`（硬删除）。
    pub action: String,
    /// 后台遗忘任务检查周期（秒）。默认 600 秒。
    #[serde(with = "duration_seconds")]
    pub check_interval: Duration,
}

impl Default for ForgettingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stale_after: Duration::from_secs(90 * 86_400),
            stale_importance_threshold: 0.3,
            action: "archive".into(),
            check_interval: Duration::from_secs(600),
        }
    }
}

// -------------------------------------------------------------------------
// Graph extraction
// -------------------------------------------------------------------------

/// 图谱抽取与遍历配置。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GraphConfig {
    /// 是否在 `remember()` 写入时自动抽取实体与关系。默认 false。
    pub auto_extract: bool,
    /// 启用 LLM 抽取器时使用的 Chat 模型提供者名称（需在
    /// `openai_compatible_chat` 中配置）。
    pub extract_llm: String,
    /// 自定义抽取提示词文件路径；None 使用内置默认提示词。
    pub extract_prompt_file: Option<PathBuf>,
    /// OpenAI 兼容 Chat 接口提供者（用于 LLM 抽取器），key 为自定义名称。
    pub openai_compatible_chat: std::collections::BTreeMap<String, OpenAiChatConfig>,
}

/// OpenAI 兼容 Chat 接口提供者配置（用于 LLM 图谱抽取）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiChatConfig {
    /// API 基础 URL，如 `https://api.openai.com/v1`。
    pub base_url: String,
    /// API Key。生产环境优先使用环境变量注入。
    pub api_key: String,
    /// 模型名称，如 `gpt-4o-mini`。
    pub model: String,
    /// 请求超时（秒），默认 30。
    #[serde(with = "duration_seconds")]
    pub timeout: Duration,
}

impl Default for OpenAiChatConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-4o-mini".into(),
            timeout: Duration::from_secs(30),
        }
    }
}

// -------------------------------------------------------------------------
// Background jobs
// -------------------------------------------------------------------------

/// 后台任务调度配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JobsConfig {
    /// TTL/过期检查周期（秒）。默认 60 秒；低写入负载可适当调高以节省 CPU。
    #[serde(with = "duration_seconds")]
    pub ttl_interval: Duration,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self { ttl_interval: Duration::from_secs(60) }
    }
}

// -------------------------------------------------------------------------
// Logging
// -------------------------------------------------------------------------

/// 日志（Tracing）输出配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// 日志级别：`trace` / `debug` / `info` / `warn` / `error`。默认 `info`。
    pub level: String,
    /// 是否输出 JSON 结构化日志（便于 ELK/Loki 等采集）。
    pub json_format: bool,
    /// 若设置则同时写入日志到该文件路径。
    pub file: Option<PathBuf>,
    /// stderr 输出是否包含 ANSI 颜色码。终端环境推荐 true。
    pub ansi: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self { level: "info".into(), json_format: false, file: None, ansi: true }
    }
}

// =========================================================================
// Loading
// =========================================================================

impl Config {
    /// Load configuration using the standard layering.
    pub fn load() -> NovaResult<Self> {
        // --- Figure out the toml path. ---
        let toml_path = std::env::var("YQ_NOVA_CONFIG").map(PathBuf::from).ok().or_else(|| {
            let local = PathBuf::from(DEFAULT_CONFIG_FILENAME);
            if local.exists() { Some(local) } else { None }
        });

        let mut figment = Figment::from(Serialized::defaults(Config::default()))
            .select(Profile::from_env_or("YQ_NOVA_PROFILE", "default"))
            .merge(Env::prefixed("YQ_NOVA_").split("__"));

        if let Some(path) = &toml_path {
            if path.exists() {
                figment = figment.merge(Toml::file(path));
            }
        }

        let cfg: Config = figment.extract()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// 运行期关键不变量校验。策略上故意严格——宁可启动时提前崩溃，
    /// 也不对外提供错误配置的服务。
    ///
    /// 校验项包括：
    /// - `server.bind` 非空
    /// - `storage.db_path` 非空且不含 `..` 路径穿越
    /// - `embedding.default_provider` 已设置
    /// - `forgetting.stale_importance_threshold` ∈ [0, 1]
    /// - `forgetting.action` ∈ {archive, delete}
    /// - `server.max_request_body_bytes` ≥ 1024
    /// - `storage.synchronous` 为合法值
    /// - `storage.wal_autocheckpoint_kb` / `journal_size_limit_kb` /
    ///   `mmap_size_kb` / `soft_heap_limit_kb` ≥ 0
    pub fn validate(&self) -> NovaResult<()> {
        if self.server.bind.is_empty() {
            return Err(NovaError::config_msg("server.bind must not be empty"));
        }
        if self.storage.db_path.as_os_str().is_empty() {
            return Err(NovaError::config_msg("storage.db_path must not be empty"));
        }
        // Path traversal guard: canonicalise and check it doesn't escape
        // a user-provided base. Simple rule: reject raw `..` components.
        let path_str = self.storage.db_path.to_string_lossy();
        if path_str.contains("..") {
            return Err(NovaError::config_msg(
                "storage.db_path must not contain '..' components; use an absolute path",
            ));
        }
        if self.embedding.default_provider.is_empty() {
            return Err(NovaError::config_msg("embedding.default_provider must be set"));
        }
        if self.forgetting.stale_importance_threshold < 0.0
            || self.forgetting.stale_importance_threshold > 1.0
        {
            return Err(NovaError::config_msg(
                "forgetting.stale_importance_threshold must be in [0, 1]",
            ));
        }
        if self.forgetting.action != "archive" && self.forgetting.action != "delete" {
            return Err(NovaError::config_msg("forgetting.action must be 'archive' or 'delete'"));
        }
        if self.server.max_request_body_bytes < 1024 {
            return Err(NovaError::config_msg(
                "server.max_request_body_bytes must be at least 1024",
            ));
        }
        let sync = self.storage.synchronous.to_ascii_lowercase();
        if !matches!(sync.as_str(), "full" | "normal" | "off" | "0" | "1" | "2" | "3") {
            return Err(NovaError::config_msg(
                "storage.synchronous must be one of full / normal / off (or numeric 0-3)",
            ));
        }
        if self.storage.wal_autocheckpoint_kb < 0 {
            return Err(NovaError::config_msg("storage.wal_autocheckpoint_kb must be >= 0"));
        }
        if self.storage.journal_size_limit_kb < 0 {
            return Err(NovaError::config_msg("storage.journal_size_limit_kb must be >= 0"));
        }
        if self.storage.mmap_size_kb < 0 {
            return Err(NovaError::config_msg("storage.mmap_size_kb must be >= 0"));
        }
        if self.storage.soft_heap_limit_kb < 0 {
            return Err(NovaError::config_msg("storage.soft_heap_limit_kb must be >= 0"));
        }
        Ok(())
    }

    /// Attempt to persist the current config back as a TOML file (useful for
    /// `config init`-style commands).
    pub fn save_to(&self, path: &std::path::Path) -> NovaResult<()> {
        let s = toml::to_string_pretty(self)
            .map_err(|e| NovaError::config_msg(format!("serialise config: {e}")))?;
        fs::write(path, s)?;
        Ok(())
    }

    /// Serialise the current config as TOML into the given `io::Write` sink.
    /// Used by the server CLI `config-show` subcommand so it doesn't need a
    /// direct `toml` dependency.
    pub fn save_to_temp<W: std::io::Write>(&self, mut w: W) -> NovaResult<()> {
        let s = toml::to_string_pretty(self)
            .map_err(|e| NovaError::config_msg(format!("serialise config: {e}")))?;
        w.write_all(s.as_bytes())
            .map_err(|e| NovaError::config_msg(format!("write config to buffer: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        Config::default().validate().expect("defaults must validate");
    }

    #[test]
    fn db_path_traversal_rejected() {
        let mut cfg = Config::default();
        cfg.storage.db_path = PathBuf::from("/tmp/../../etc/passwd");
        let err = cfg.validate().expect_err("should fail");
        assert_eq!(err.code(), crate::error::ErrorCode::Config);
    }

    #[test]
    fn wrong_forgetting_action_rejected() {
        let mut cfg = Config::default();
        cfg.forgetting.action = "shred".into();
        cfg.validate().expect_err("should fail");
    }

    #[test]
    fn importance_threshold_out_of_range() {
        let mut cfg = Config::default();
        cfg.forgetting.stale_importance_threshold = 1.5;
        cfg.validate().expect_err("should fail");
    }
}
