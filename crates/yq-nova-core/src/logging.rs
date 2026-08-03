//! Logging / tracing subsystem.
//!
//! Wraps [`tracing_subscriber`] with dual stdout + optional file output,
//! JSON or human-readable formatting, and a per-request `trace_id`
//! generated via UUID v4.

use std::{path::Path, sync::OnceLock};

use tracing::Dispatch;
use tracing_subscriber::{
    EnvFilter, Layer,
    filter::{Directive, LevelFilter},
    fmt::{self, time::UtcTime},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

use crate::{
    config::LoggingConfig,
    error::{NovaError, NovaResult},
};

/// Global `trace_id` — cheap `OnceLock`; populated by the first call to
/// [`current_trace_id`]. Not intended to be used by request paths (which
/// generate per-request ids via tower layer).
static GLOBAL_TRACE_ID: OnceLock<String> = OnceLock::new();

/// Lazy initialisation guard for `tracing`. We use `set_global_default` so
/// `log` crate calls are also bridged (via `tracing-log`).
static INIT: OnceLock<()> = OnceLock::new();

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Initialise the global tracing subscriber from the supplied config.
///
/// Idempotent: only the first call has any effect; subsequent calls are
/// no-ops (return `Ok(false)`). This lets us call `init` from tests and
/// from `main` without worrying about double-init panics.
pub fn init_tracing(cfg: &LoggingConfig) -> NovaResult<bool> {
    if INIT.get().is_some() {
        return Ok(false);
    }

    let filter = build_env_filter(cfg);
    let timer = UtcTime::rfc_3339();

    // --- stderr / stdout layer (human or JSON). ---
    let fmt_layer = fmt::layer()
        .with_timer(timer.clone())
        .with_target(true)
        .with_line_number(true)
        .with_ansi(cfg.ansi)
        .with_writer(std::io::stderr);

    let fmt_layer = if cfg.json_format {
        fmt_layer.json().with_current_span(false).boxed()
    } else {
        (fmt_layer).boxed()
    };

    // --- Optional file layer, always JSON if enabled. ---
    let file_layer = if let Some(file) = &cfg.file {
        let non_blocking = make_non_blocking_file_writer(file)?;
        let layer = fmt::layer()
            .with_timer(timer)
            .with_target(true)
            .with_line_number(true)
            .with_ansi(false)
            .json()
            .with_current_span(false)
            .with_writer(non_blocking);
        Some((layer).boxed())
    } else {
        None
    };

    let registry = tracing_subscriber::registry().with(filter).with(fmt_layer);
    if let Some(file) = file_layer {
        registry.with(file).try_init().map_err(|e| {
            NovaError::config_msg(format!("failed to init tracing subscriber: {e}"))
        })?;
    } else {
        registry.try_init().map_err(|e| {
            NovaError::config_msg(format!("failed to init tracing subscriber: {e}"))
        })?;
    }

    INIT.get_or_init(|| ());

    // Bootstrap a global trace-id so early startup logs can be correlated.
    let _ = current_trace_id();
    Ok(true)
}

/// Return (and lazily create) a process-wide `trace_id` string.
///
/// Individual HTTP requests replace this with their own span-level id;
/// this fallback is intended only for startup / background jobs.
pub fn current_trace_id() -> &'static str {
    GLOBAL_TRACE_ID.get_or_init(|| uuid::Uuid::new_v4().to_string())
}

/// Construct a [`Dispatch`] handle used for per-task log routing (rarely
/// needed). Mostly exposed for tests.
pub fn current_dispatch() -> Dispatch {
    tracing::dispatcher::get_default(|d| d.clone())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_env_filter(cfg: &LoggingConfig) -> EnvFilter {
    // Start from `YQ_NOVA_LOG` or `RUST_LOG` if set; otherwise from the
    // config level string, falling back to a sensible default directive.
    let from_env = std::env::var_os("YQ_NOVA_LOG")
        .or_else(|| std::env::var_os("RUST_LOG"))
        .and_then(|s| s.into_string().ok());

    let base: Directive = if let Some(s) = from_env {
        match s.parse::<Directive>() {
            Ok(d) => d,
            Err(_) => parse_level(&cfg.level),
        }
    } else {
        parse_level(&cfg.level)
    };

    EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
        .add_directive(base)
        // Silence noisy upstream crates by default regardless of user level.
        .add_directive("hyper=warn".parse().unwrap())
        .add_directive("rustls=warn".parse().unwrap())
        .add_directive("reqwest=warn".parse().unwrap())
        .add_directive("sqlx::query=warn".parse().unwrap())
}

fn parse_level(s: &str) -> Directive {
    match s.to_ascii_lowercase().as_str() {
        "trace" => LevelFilter::TRACE.into(),
        "debug" => LevelFilter::DEBUG.into(),
        "warn" => LevelFilter::WARN.into(),
        "error" => LevelFilter::ERROR.into(),
        "off" => LevelFilter::OFF.into(),
        _ => LevelFilter::INFO.into(),
    }
}

fn make_non_blocking_file_writer(
    path: &Path,
) -> NovaResult<tracing_appender::non_blocking::NonBlocking> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                NovaError::config_msg(format!("create log dir {}: {e}", parent.display()))
            })?;
        }
    }
    let file_appender = tracing_appender::rolling::never(
        path.parent().unwrap_or_else(|| Path::new(".")),
        path.file_name().unwrap_or_else(|| std::ffi::OsStr::new("yq-nova.log")),
    );
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    // Leak the guard on purpose: we want the non-blocking writer to live
    // for the entire process lifetime; flushes happen on drop but we rely
    // on the process OS flush anyway.
    std::mem::forget(_guard);
    Ok(non_blocking)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_aliases() {
        // idempotent init, no-op if already set up.
        let _ = init_tracing(&LoggingConfig::default());
        assert_eq!(parse_level("TRACE"), Directive::from(LevelFilter::TRACE));
        assert_eq!(parse_level("error"), Directive::from(LevelFilter::ERROR));
        assert_eq!(parse_level("nonexistent"), Directive::from(LevelFilter::INFO));
    }

    #[test]
    fn trace_id_is_stable() {
        let a = current_trace_id();
        let b = current_trace_id();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }
}
