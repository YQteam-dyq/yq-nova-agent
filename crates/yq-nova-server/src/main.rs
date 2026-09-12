use std::{net::SocketAddr, process::ExitCode, sync::Arc, time::Duration};

use clap::{Args, Parser, Subcommand, ValueEnum, builder::TypedValueParser};
use tracing::{error, info, warn};
use yq_nova_core::{
    VERSION,
    config::Config,
    error::NovaResult,
    graph::{GraphService, extractor::RegexWikiExtractor},
    logging,
    memory::{
        ForgetMode, MemoryService, SearchMode, ops_forget, ops_list, ops_recall, ops_remember,
        ops_tag,
    },
    storage::{Database, MemoryFilter, MemorySortOrder, MemorySource, parse_sources, parse_statuses},
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
    #[arg(short, long, env = "YQ_NOVA_CONFIG")]
    config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Serve,

    Check,

    ConfigShow,

    InitDb,

    Remember(RememberArgs),

    Recall(RecallArgs),

    Forget(ForgetArgs),

    Stats,

    List(ListArgs),

    Tags(TagsArgs),
}

#[derive(Debug, Args)]
struct RememberArgs {
    #[arg(required_unless_present = "content")]
    content_pos: Option<String>,

    #[arg(long)]
    content: Option<String>,

    #[arg(long, value_parser = clap_num())]
    importance: Option<f32>,

    #[arg(long, default_value_t = MemorySourceCli::User)]
    source: MemorySourceCli,

    #[arg(long = "tag")]
    tags: Vec<String>,

    #[arg(long, default_value_t = true)]
    embed: bool,

    #[arg(long, default_value_t = false)]
    extract_graph: bool,

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
    query: String,

    #[arg(long, default_value_t = 10)]
    top_k: usize,

    #[arg(long, value_enum, default_value_t = SearchModeCli::Semantic)]
    mode: SearchModeCli,

    #[arg(long, default_value_t = 0.0)]
    score_threshold: f32,

    #[arg(long, default_value_t = false)]
    graph: bool,

    #[arg(long, default_value_t = 2)]
    graph_depth: u8,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ForgetArgs {
    #[arg(long)]
    uuid: Option<String>,

    #[arg(long = "tag-all")]
    tag_all: Vec<String>,

    #[arg(long)]
    importance_max: Option<f32>,

    #[arg(long, value_enum, default_value_t = ForgetModeCli::Soft)]
    mode: ForgetModeCli,

    #[arg(long, default_value_t = false)]
    gc_graph: bool,

    #[arg(long, default_value_t = 1000)]
    batch_limit: usize,

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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SortOrderCli {
    CreatedDesc,
    CreatedAsc,
    ImportanceDesc,
    ImportanceAsc,
    AccessedDesc,
}
impl std::fmt::Display for SortOrderCli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(MemorySortOrder::from(*self).as_str())
    }
}
impl From<SortOrderCli> for MemorySortOrder {
    fn from(v: SortOrderCli) -> Self {
        match v {
            SortOrderCli::CreatedDesc => MemorySortOrder::CreatedDesc,
            SortOrderCli::CreatedAsc => MemorySortOrder::CreatedAsc,
            SortOrderCli::ImportanceDesc => MemorySortOrder::ImportanceDesc,
            SortOrderCli::ImportanceAsc => MemorySortOrder::ImportanceAsc,
            SortOrderCli::AccessedDesc => MemorySortOrder::AccessedDesc,
        }
    }
}

#[derive(Debug, Args)]
struct ListArgs {
    #[arg(long, default_value_t = 20)]
    limit: u32,

    #[arg(long, default_value_t = 0)]
    offset: u32,

    #[arg(long, value_enum, default_value_t = SortOrderCli::CreatedDesc)]
    sort: SortOrderCli,

    #[arg(long = "tag")]
    tags: Vec<String>,

    #[arg(long)]
    status: Option<String>,

    #[arg(long)]
    source: Option<String>,

    #[arg(long)]
    importance_min: Option<f32>,

    #[arg(long)]
    importance_max: Option<f32>,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct TagsArgs {
    #[arg(long, default_value_t = 50)]
    limit: u32,

    #[arg(long, default_value_t = 0)]
    offset: u32,

    #[arg(long)]
    json: bool,
}

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
    let exit_code = match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fatal: {e:#}");
            error!(error = %e, "yq-nova exited with error");
            ExitCode::FAILURE
        },
    };

    #[cfg(feature = "otel")]
    yq_nova_core::logging::shutdown_otel();

    exit_code
}

async fn run() -> NovaResult<()> {
    let cli = Cli::parse();
    if let Some(path) = &cli.config {
        std::env::set_var("YQ_NOVA_CONFIG", path);
    }

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
                chunk_options: None,
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
                group_chunks: false,
                entity_focus: Vec::new(),
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
                    "affected_memories={aff} cascade_embeddings={casc} gc_entities={ge} \
                     gc_relations={gr}",
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
        Commands::List(args) => {
            let (db, memory, _graph, _provider_name) = open_core_services(&cfg).await?;
            let status_in = match args.status.as_deref() {
                Some(raw) => Some(parse_statuses(raw)?),
                None => None,
            };
            let source_in = match args.source.as_deref() {
                Some(raw) => Some(parse_sources(raw)?),
                None => None,
            };
            let filter = MemoryFilter {
                status_in,
                source_in,
                importance_min: args.importance_min,
                importance_max: args.importance_max,
                tags_all: if args.tags.is_empty() { None } else { Some(args.tags) },
                ..Default::default()
            };
            let out = memory
                .list_memories(ops_list::ListInput {
                    filter,
                    limit: args.limit,
                    offset: args.offset,
                    sort: args.sort.into(),
                })
                .await?;
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out)
                        .map_err(|e| yq_nova_core::NovaError::internal(e.to_string()))?
                );
            } else {
                print_memory_list(&out);
            }
            db.close().await
        },
        Commands::Tags(args) => {
            let (db, memory, _graph, _provider_name) = open_core_services(&cfg).await?;
            let out = memory
                .list_tags(ops_tag::TagListInput {
                    limit: args.limit,
                    offset: args.offset,
                })
                .await?;
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out)
                        .map_err(|e| yq_nova_core::NovaError::internal(e.to_string()))?
                );
            } else {
                print_tags(&out);
            }
            db.close().await
        },
        Commands::Serve => {
            let db = Database::open(cfg.storage.clone()).await?;
            info!(size_bytes = db.size_on_disk_bytes()?, "database ready");

            let (provider_name, provider, embed_dims, _registry) =
                provider_wiring::build_registry(&cfg.embedding)?;
            info!(
                provider = %provider_name,
                embed_dims,
                "embedding provider ready"
            );
            if provider_name == "mock" {
                tracing::warn!(
                    "embedding provider is 'mock' — semantic search will be deterministic only by \
                     importance/access, not semantics. Set YQ_NOVA_EMBEDDING__DEFAULT_PROVIDER or \
                     config.embedding.default_provider to a real name for production."
                );
            }

            let memory = MemoryService::new(db.clone(), provider.clone());
            let graph = GraphService::with_parts(db.clone(), Arc::new(RegexWikiExtractor::new()));

            let cancel = background::new_cancel_token();
            let job_cancel = cancel.clone();
            let memory_for_jobs = memory.clone();
            let forgetting_cfg = cfg.forgetting.clone();

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

            let _ = Duration::from_secs(60);
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = wait_for_shutdown_signal().await;
                    info!("graceful shutdown: draining in-flight requests & stopping jobs");
                    cancel.cancel();
                })
                .await
                .map_err(|e| yq_nova_core::NovaError::internal_with_ctx("axum serve", e))?;

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

            info!("flushing SQLite WAL and closing pool");
            db.close().await?;
            Ok(())
        },
    }
}

fn print_memory_list(out: &ops_list::ListOutput) {
    println!(
        "total={total} count={count} limit={limit} offset={offset} sort={sort}",
        total = out.total,
        count = out.count,
        limit = out.limit,
        offset = out.offset,
        sort = out.sort.as_str(),
    );
    for (i, m) in out.items.iter().enumerate() {
        let snippet: String = m.content.chars().take(80).collect();
        println!(
            "  #{idx:>3}: imp={imp:.2} acc={acc:<4} status={st:<8} tags={tags:?} {snip}",
            idx = i + 1,
            imp = m.importance,
            acc = m.access_count,
            st = m.status,
            tags = m.tags,
            snip = snippet,
        );
    }
}

fn print_tags(out: &ops_tag::TagListOutput) {
    println!("total={total} count={count}", total = out.total, count = out.count);
    for t in &out.items {
        println!("  {name:<32} memories={count}", name = t.name, count = t.memory_count);
    }
}

async fn open_core_services(
    cfg: &Config,
) -> NovaResult<(Database, MemoryService, GraphService, String)> {
    let db = Database::open(cfg.storage.clone()).await?;
    let (provider_name, provider, _embed_dims, _registry) =
        provider_wiring::build_registry(&cfg.embedding)?;
    if provider_name == "mock" {
        warn!(
            "embedding provider is 'mock' — semantic search will be deterministic only by \
             importance/access, not semantics."
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
