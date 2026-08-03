//! yq-nova — lightweight Agent memory & state layer (binary entrypoint).
//!
//! Minimum behaviour implemented at M4:
//!   1. Parse `Config` from env / toml
//!   2. Initialise tracing
//!   3. Open / migrate the SQLite database
//!   4. Start axum HTTP server on `server.bind`
//!   5. Graceful shutdown on SIGINT/SIGTERM
//!
//! TODO list for later milestones lives in ../../tasks.md

use std::{net::SocketAddr, process::ExitCode, sync::Arc, time::Duration};

use clap::builder::TypedValueParser;
use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing::{error, info, warn};

use yq_nova_core::{
    VERSION,
    config::Config,
    error::NovaResult,
    graph::GraphService,
    graph::extractor::RegexWikiExtractor,
    logging,
    memory::{ForgetMode, MemoryService, SearchMode, ops_forget, ops_recall, ops_remember},
    storage::{Database, MemorySource},
};

mod background;
mod http;
mod provider_wiring;

use crate::http::{AppState, build_router};

#[derive(Debug, Parser)]
#[command(
    name = "yq-nova",
    version = VERSION,
    about = "Lightweight, single-file Agent memory & state layer",
    long_about = None,
)]
struct Cli {
    /// Path to a TOML config file (overrides YQ_NOVA_CONFIG env var).
    #[arg(short, long, env = "YQ_NOVA_CONFIG")]
    config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run the HTTP server (default if no subcommand is given).
    Serve,
    /// Validate the current config and SQLite file, then exit.
    Check,
    /// Dump the merged, validated config as TOML to stdout. Useful for
    /// debugging env-var / toml layering.
    ConfigShow,
    /// Initialise the DB (create + migrate) then exit. Good for CI/CD.
    InitDb,
    /// Remember a single piece of content directly (no HTTP).
    Remember(RememberArgs),
    /// Run a recall query and print hits (no HTTP).
    Recall(RecallArgs),
    /// Forget memories by UUID or filter (no HTTP).
    Forget(ForgetArgs),
    /// Print DB stats (active/archived counts, entities, relations, DB size).
    Stats,
}

#[derive(Debug, Args)]
struct RememberArgs {
    /// Memory content (required). Pass as positional arg or via --content.
    #[arg(required_unless_present = "content")]
    content_pos: Option<String>,
    /// Alternatively, pass content via this flag.
    #[arg(long)]
    content: Option<String>,
    /// Importance in `[0, 1]`. Default 0.5.
    #[arg(long, value_parser = clap_num())]
    importance: Option<f32>,
    /// Memory source label.
    #[arg(long, default_value_t = MemorySourceCli::User)]
    source: MemorySourceCli,
    /// Attach a tag (repeatable).
    #[arg(long = "tag")]
    tags: Vec<String>,
    /// Compute + store the semantic embedding.
    #[arg(long, default_value_t = true)]
    embed: bool,
    /// Run WikiLink/#tag graph extraction and attach entities/relations.
    #[arg(long, default_value_t = false)]
    extract_graph: bool,
    /// Output as JSON instead of a human-friendly one-liner.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MemorySourceCli {
    Agent,
    User,
    System,
    Tool,
}
impl std::fmt::Display for MemorySourceCli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MemorySourceCli::Agent => "agent",
            MemorySourceCli::User => "user",
            MemorySourceCli::System => "system",
            MemorySourceCli::Tool => "tool",
        })
    }
}
impl From<MemorySourceCli> for MemorySource {
    fn from(v: MemorySourceCli) -> Self {
        match v {
            MemorySourceCli::Agent => MemorySource::Agent,
            MemorySourceCli::User => MemorySource::User,
            MemorySourceCli::System => MemorySource::System,
            MemorySourceCli::Tool => MemorySource::Tool,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SearchModeCli {
    Semantic,
    Keyword,
    Hybrid,
}
impl std::fmt::Display for SearchModeCli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SearchModeCli::Semantic => "semantic",
            SearchModeCli::Keyword => "keyword",
            SearchModeCli::Hybrid => "hybrid",
        })
    }
}
impl From<SearchModeCli> for SearchMode {
    fn from(v: SearchModeCli) -> Self {
        match v {
            SearchModeCli::Semantic => SearchMode::Semantic,
            SearchModeCli::Keyword => SearchMode::Keyword,
            SearchModeCli::Hybrid => SearchMode::Hybrid,
        }
    }
}

#[derive(Debug, Args)]
struct RecallArgs {
    /// Query string.
    query: String,
    /// Number of hits to return.
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    /// Retrieval mode.
    #[arg(long, value_enum, default_value_t = SearchModeCli::Semantic)]
    mode: SearchModeCli,
    /// Minimum final-score threshold in `[0,1]`.
    #[arg(long, default_value_t = 0.0)]
    score_threshold: f32,
    /// Enable graph expansion for this recall.
    #[arg(long, default_value_t = false)]
    graph: bool,
    /// Max BFS depth when --graph is on.
    #[arg(long, default_value_t = 2)]
    graph_depth: u8,
    /// Print each hit as JSON lines instead of a human table.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ForgetArgs {
    /// Forget one specific UUID.
    #[arg(long)]
    uuid: Option<String>,
    /// Forget by tag name (repeatable; ALL tags must match).
    #[arg(long = "tag-all")]
    tag_all: Vec<String>,
    /// Maximum importance a row can have to be eligible for forgetting.
    #[arg(long)]
    importance_max: Option<f32>,
    /// `soft` (archive) or `hard` (delete).
    #[arg(long, value_enum, default_value_t = ForgetModeCli::Soft)]
    mode: ForgetModeCli,
    /// Also cascade-orphan dangling graph entities (UUID path only).
    #[arg(long, default_value_t = false)]
    gc_graph: bool,
    /// Safety cap on rows affected by a filter-based forget.
    #[arg(long, default_value_t = 1000)]
    batch_limit: usize,
    /// Print a summary JSON blob.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ForgetModeCli {
    Soft,
    Hard,
}
impl std::fmt::Display for ForgetModeCli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ForgetModeCli::Soft => "soft",
            ForgetModeCli::Hard => "hard",
        })
    }
}
impl From<ForgetModeCli> for ForgetMode {
    fn from(v: ForgetModeCli) -> Self {
        match v {
            ForgetModeCli::Soft => ForgetMode::Archive,
            ForgetModeCli::Hard => ForgetMode::Hard,
        }
    }
}

// Helper so `#[arg(value_parser = clap_num())]` works for `f32` in 0..=1.
fn clap_num() -> impl clap::builder::TypedValueParser<Value = f32> {
    clap::builder::StringValueParser::new().try_map(|s: String| {
        let n: f32 = s.parse::<f32>().map_err(|e| format!("expected float: {e}"))?;
        if !(0.0_f32..=1.0).contains(&n) {
            return Err(format!("value must be in [0, 1], got {n}"));
        }
        Ok(n)
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // If tracing isn't initialised yet we still want a readable error.
            eprintln!("fatal: {e:#}");
            error!(error = %e, "yq-nova exited with error");
            ExitCode::FAILURE
        },
    }
}

async fn run() -> NovaResult<()> {
    // ------- CLI parse + optional override of config path -------
    let cli = Cli::parse();
    if let Some(path) = &cli.config {
        std::env::set_var("YQ_NOVA_CONFIG", path);
    }

    // ------- Load & validate config, init logging -------
    let cfg = Config::load()?;
    let _ = logging::init_tracing(&cfg.logging);

    info!(
        version = VERSION,
        git_sha = yq_nova_core::git_sha(),
        bind = %cfg.server.bind,
        db_path = %cfg.storage.db_path.display(),
        "yq-nova starting"
    );

    let cmd = cli.command.unwrap_or(Commands::Serve);
    match cmd {
        Commands::ConfigShow => {
            // Use the core save_to helper to avoid a direct toml dep
            // mismatch (core already handles serialisation errors).
            let mut buf = Vec::new();
            cfg.save_to_temp(&mut buf)?;
            let s = String::from_utf8(buf)
                .map_err(|e| yq_nova_core::NovaError::internal(format!("toml utf8: {e}")))?;
            println!("{s}");
            Ok(())
        },
        Commands::Check => {
            info!("opening & migrating DB for dry-run validation...");
            let db = Database::open(cfg.storage.clone()).await?;
            let n = db.size_on_disk_bytes()?;
            info!(size_bytes = n, "db ok");
            eprintln!("config: valid\ndb size: {n} bytes");
            db.close().await
        },
        Commands::InitDb => {
            info!("initialising DB ...");
            let db = Database::open(cfg.storage.clone()).await?;
            info!(size_bytes = db.size_on_disk_bytes()?, "db ready");
            db.close().await
        },
        Commands::Remember(args) => {
            let (db, memory, _graph, _provider_name) = open_core_services(&cfg).await?;
            let content: String = args.content_pos.or(args.content).ok_or_else(|| {
                yq_nova_core::NovaError::validation("remember: content is required")
            })?;
            let mut tags = args.tags;
            tags.sort_unstable();
            tags.dedup();
            let input = ops_remember::RememberInput {
                content: &content,
                importance: args.importance.unwrap_or(0.5),
                source: args.source.into(),
                metadata: None,
                expires_at: None,
                tags: &tags,
                embed: args.embed,
                extract_graph: args.extract_graph,
            };
            let out = memory.remember(input).await?;
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out)
                        .map_err(|e| yq_nova_core::NovaError::internal(e.to_string()))?
                );
            } else {
                println!(
                    "uuid={uuid} duplicate={dup} emb_store={emb} entities={ent} tags={tags:?}",
                    uuid = out.uuid,
                    dup = out.duplicate,
                    emb = out.embedding_stored,
                    ent = out.entities_extracted,
                    tags = out.tags,
                );
            }
            db.close().await
        },
        Commands::Recall(args) => {
            let (db, memory, _graph, _provider_name) = open_core_services(&cfg).await?;
            let input = ops_recall::RecallInput {
                query: &args.query,
                top_k: args.top_k,
                score_threshold: args.score_threshold,
                similarity_threshold: -1.0,
                mode: args.mode.into(),
                graph: yq_nova_core::memory::GraphTraversalOpts {
                    enabled: args.graph,
                    max_depth: args.graph_depth,
                    predicate_whitelist: vec![],
                },
                hybrid_weights: None,
                rrf_k: None,
                rank_weights: None,
                filter: Default::default(),
            };
            let out = memory.recall(input).await?;
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out)
                        .map_err(|e| yq_nova_core::NovaError::internal(e.to_string()))?
                );
            } else {
                println!(
                    "hits={count} total_candidates={total} query={q}",
                    count = out.hits.len(),
                    total = out.total_candidates,
                    q = out.query,
                );
                for (i, h) in out.hits.iter().enumerate() {
                    let sim =
                        h.raw_similarity.map(|x| format!("{x:.3}")).unwrap_or_else(|| "-".into());
                    let g = if h.from_graph { "+graph" } else { "     " };
                    let snippet: String = h.memory.content.chars().take(90).collect();
                    println!(
                        "  #{i:>2}: score={s:.3} sim={sim:>5} imp={imp:.2} acc={acc:<3} {g} {snip}",
                        i = i + 1,
                        s = h.final_score,
                        sim = sim,
                        imp = h.memory.importance,
                        acc = h.memory.access_count,
                        g = g,
                        snip = snippet,
                    );
                }
            }
            db.close().await
        },
        Commands::Forget(args) => {
            use ops_forget::{ForgetInput, ForgetTarget};
            let (db, memory, _graph, _provider_name) = open_core_services(&cfg).await?;

            let target = if let Some(uuid_s) = args.uuid {
                let u = uuid::Uuid::parse_str(&uuid_s).map_err(|e| {
                    yq_nova_core::NovaError::validation_msg(format!("invalid --uuid {uuid_s}: {e}"))
                })?;
                ForgetTarget::One(u)
            } else {
                let f = yq_nova_core::storage::MemoryFilter {
                    tags_all: if args.tag_all.is_empty() { None } else { Some(args.tag_all) },
                    importance_max: args.importance_max,
                    ..Default::default()
                };
                ForgetTarget::Filter(f)
            };
            let input = ForgetInput {
                target,
                mode: args.mode.into(),
                gc_graph: args.gc_graph,
                batch_limit: args.batch_limit,
            };
            let out = memory.forget(input).await?;
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out)
                        .map_err(|e| yq_nova_core::NovaError::internal(e.to_string()))?
                );
            } else {
                println!(
                    "affected_memories={aff} cascade_embeddings={casc} gc_entities={ge} gc_relations={gr}",
                    aff = out.affected_memories,
                    casc = out.cascade_embeddings,
                    ge = out.gc_entities,
                    gr = out.gc_relations,
                );
            }
            db.close().await
        },
        Commands::Stats => {
            let (db, _memory, _graph, _provider_name) = open_core_services(&cfg).await?;

            use yq_nova_core::storage::{
                MemoryFilter, MemoryRepository, MemoryStatus, SqliteMemoryRepository,
            };
            let mem_repo = SqliteMemoryRepository::new();
            let active_count: i64 = mem_repo
                .count(
                    &db,
                    &MemoryFilter {
                        status_in: Some(vec![MemoryStatus::Active]),
                        ..Default::default()
                    },
                )
                .await
                .unwrap_or(0);
            let archived_count: i64 = mem_repo
                .count(
                    &db,
                    &MemoryFilter {
                        status_in: Some(vec![MemoryStatus::Archived]),
                        ..Default::default()
                    },
                )
                .await
                .unwrap_or(0);

            let entity_count: i64 = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM entities")
                .fetch_one(&db.pool)
                .await
                .unwrap_or(0);
            let relation_count: i64 =
                sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM relations")
                    .fetch_one(&db.pool)
                    .await
                    .unwrap_or(0);
            let db_size = db.size_on_disk_bytes().unwrap_or(0);

            println!(
                "active={active} archived={arch} entities={ent} relations={rel} db_size_bytes={db}",
                active = active_count,
                arch = archived_count,
                ent = entity_count,
                rel = relation_count,
                db = db_size,
            );
            db.close().await
        },
        Commands::Serve => {
            let db = Database::open(cfg.storage.clone()).await?;
            info!(size_bytes = db.size_on_disk_bytes()?, "database ready");

            // --- M5.1/5.2: 按配置选择 embedding provider ---
            let (provider_name, provider, embed_dims, _registry) =
                provider_wiring::build_registry(&cfg.embedding)?;
            info!(
                provider = %provider_name,
                embed_dims,
                "embedding provider ready"
            );
            if provider_name == "mock" {
                tracing::warn!(
                    "embedding provider is 'mock' — semantic search will be deterministic \
                     only by importance/access, not semantics. Set YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER \
                     or config.embedding.default_provider to a real name for production."
                );
            }

            let memory = MemoryService::new(db.clone(), provider.clone());
            let graph = GraphService::with_parts(db.clone(), Arc::new(RegexWikiExtractor::new()));

            // --- M6.3: Spawn background jobs (TTL + GC) with graceful shutdown ---
            let cancel = background::new_cancel_token();
            let job_cancel = cancel.clone();
            let memory_for_jobs = memory.clone();
            let forgetting_cfg = cfg.forgetting.clone();
            // TTL job runs at a fixed short cadence; GC cadence is driven by
            // forgetting_cfg.check_interval (slower). Combine them by letting
            // the faster ticker wake up; GC simply early-returns on the ticks
            // that are off-phase. We use the ttl_interval for now (it's the
            // fast one). Later can split into two independent handles if we
            // want precise per-job intervals.
            let _forgetting_owned = forgetting_cfg.clone();
            let jobs_handle = background::spawn_job_loop(
                memory_for_jobs,
                forgetting_cfg,
                cfg.jobs.ttl_interval,
                job_cancel,
            );

            let state = AppState::new(cfg.server.clone(), db.clone(), memory, graph);
            let router = build_router(state);

            let addr: SocketAddr = cfg.server.bind.parse().map_err(|e| {
                yq_nova_core::NovaError::config_msg(format!(
                    "invalid server.bind {}: {e}",
                    cfg.server.bind
                ))
            })?;
            info!(%addr, "yq-nova HTTP server starting");

            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| yq_nova_core::NovaError::internal_with_ctx("bind tcp", e))?;

            // TCP keepalive for the server sockets.
            let _ = Duration::from_secs(60); // keep reference for future fine-tuning.
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = wait_for_shutdown_signal().await;
                    info!("graceful shutdown: draining in-flight requests & stopping jobs");
                    cancel.cancel();
                })
                .await
                .map_err(|e| yq_nova_core::NovaError::internal_with_ctx("axum serve", e))?;

            // Await the jobs handle after HTTP exits so all bookkeeping logs
            // (including final stats) flush before we exit. Put a 30s ceiling
            // so an accidentally-stuck in-flight job doesn't prevent shutdown.
            let _stats = match tokio::time::timeout(Duration::from_secs(30), jobs_handle).await {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    error!(error = %e, "job loop panicked during join");
                    Default::default()
                },
                Err(_elapsed) => {
                    warn!("jobs did not exit within 30s, proceeding with shutdown anyway");
                    Default::default()
                },
            };
            info!(
                ttl_expired = _stats.ttl_expired,
                stale_archived = _stats.stale_archived,
                stale_deleted = _stats.stale_deleted,
                "server stopped cleanly"
            );

            // Flush WAL + close SQLite pool before returning so the on-disk
            // state is fully consistent (checkpoint truncate + pool drop is
            // required to avoid leaving -wal/-shm behind after clean exit).
            info!("flushing SQLite WAL and closing pool");
            db.close().await?;
            Ok(())
        },
    }
}

// ---------------------------------------------------------------------------
// Shared helper for CLI-only subcommands: open DB + wire providers.
// ---------------------------------------------------------------------------

async fn open_core_services(
    cfg: &Config,
) -> NovaResult<(Database, MemoryService, GraphService, String)> {
    let db = Database::open(cfg.storage.clone()).await?;
    let (provider_name, provider, _embed_dims, _registry) =
        provider_wiring::build_registry(&cfg.embedding)?;
    if provider_name == "mock" {
        warn!(
            "embedding provider is 'mock' — semantic search will be deterministic \
             only by importance/access, not semantics."
        );
    }
    let memory = MemoryService::new(db.clone(), provider.clone());
    let graph = GraphService::with_parts(db.clone(), Arc::new(RegexWikiExtractor::new()));
    Ok((db, memory, graph, provider_name))
}

async fn wait_for_shutdown_signal() -> NovaResult<()> {
    use tokio::signal;

    let ctrl_c = async { signal::ctrl_c().await.map_err(|e| (e, "ctrl_c")) };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .map_err(|e| (e, "sigterm register"))?
            .recv()
            .await
            .ok_or_else(|| (std::io::Error::other("sigterm stream closed"), "sigterm"))
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<Result<_, _>>();

    tokio::select! {
        r = ctrl_c => { r.map(|_| ()).map_err(|(e, ctx)| yq_nova_core::NovaError::internal_with_ctx(ctx, e)) }
        r = terminate => { r.map(|_| ()).map_err(|(e, ctx)| yq_nova_core::NovaError::internal_with_ctx(ctx, e)) }
    }
}
