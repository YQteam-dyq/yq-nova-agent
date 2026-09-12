use std::{path::Path, sync::OnceLock};

#[cfg(feature = "otel")]
use opentelemetry::trace::TracerProvider as _;
#[cfg(feature = "otel")]
use opentelemetry_otlp::WithExportConfig;
use tracing::Dispatch;
#[cfg(feature = "otel")]
use tracing_subscriber::layer::Identity;
#[cfg(feature = "otel")]
use tracing_subscriber::registry::LookupSpan;
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

static GLOBAL_TRACE_ID: OnceLock<String> = OnceLock::new();

static INIT: OnceLock<()> = OnceLock::new();

pub fn init_tracing(cfg: &LoggingConfig) -> NovaResult<bool> {
    if INIT.get().is_some() {
        return Ok(false);
    }

    let filter = build_env_filter(cfg);
    let timer = UtcTime::rfc_3339();

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

    #[cfg(feature = "otel")]
    let registry = registry.with(init_otel_layer(cfg)?);

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

    let _ = current_trace_id();
    Ok(true)
}

pub fn current_trace_id() -> &'static str {
    GLOBAL_TRACE_ID.get_or_init(|| uuid::Uuid::new_v4().to_string())
}

pub fn current_dispatch() -> Dispatch {
    tracing::dispatcher::get_default(|d| d.clone())
}

#[cfg(feature = "otel")]
pub fn init_otel_layer<S>(cfg: &LoggingConfig) -> NovaResult<Box<dyn Layer<S> + Send + Sync>>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a> + Send + Sync,
{
    if !cfg.otel_enabled {
        return Ok(Box::new(Identity::default()));
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(cfg.otel_endpoint.clone())
        .build()
        .map_err(|e| NovaError::config_msg(format!("failed to build OTLP span exporter: {e}")))?;

    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_sampler(opentelemetry_sdk::trace::Sampler::TraceIdRatioBased(
            cfg.otel_sample_rate as f64,
        ))
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .build();

    let tracer = provider.tracer(cfg.otel_service_name.clone());
    opentelemetry::global::set_tracer_provider(provider);

    let layer = tracing_opentelemetry::layer().with_tracer(tracer).boxed();
    Ok(layer)
}

#[cfg(feature = "otel")]
pub fn shutdown_otel() {
    opentelemetry::global::shutdown_tracer_provider();
}

fn build_env_filter(cfg: &LoggingConfig) -> EnvFilter {
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

    std::mem::forget(_guard);
    Ok(non_blocking)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_aliases() {
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
