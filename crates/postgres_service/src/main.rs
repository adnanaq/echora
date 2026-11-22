//! Entry point for the Echora PostgreSQL gRPC service.

mod anime_mapping;
mod anime_relations_mapping;
mod anime_schedule_and_stats_mapping;
mod character_mapping;
mod episode_mapping;
mod service;

use anyhow::Context as _;
use common::{CommonSettings, init_telemetry};
use mimalloc::MiMalloc;
use postgres_db::{connect_pool, run_migrations};
use tokio::signal::unix::{Signal, SignalKind, signal};
use tonic::codec::CompressionEncoding;
use tonic::transport::Server;
use tracing::Level;

use service::postgres_service_proto::v1::postgres_service_server::PostgresServiceServer;
use service::{MAX_PAGE_MESSAGE_BYTES, PostgresGrpcService, read_providers_without_review};

/// Uses mimalloc, which serves the service's many small allocations faster than the system allocator.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Starts the PostgreSQL Service's gRPC server and runs until `SIGINT` or `SIGTERM`.
///
/// # Errors
///
/// Returns an error when settings, telemetry, the signal handlers, the database or the
/// server fail to start.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let settings = CommonSettings::from_env()?;
    let _telemetry_guard = init_telemetry("postgres-service", &settings)?;
    let listen_addr = settings.postgres_service_addr()?;
    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL must be set for postgres_service")?;
    let interrupt_signal = signal(SignalKind::interrupt())?;
    let terminate_signal = signal(SignalKind::terminate())?;

    let pool = connect_pool(&database_url, 10).await?;
    run_migrations(&pool).await?;

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<tonic_health::pb::health_server::HealthServer<tonic_health::server::HealthReporter>>()
        .await;

    let providers_without_review =
        read_providers_without_review(|setting_name| std::env::var(setting_name).ok());
    let grpc_service =
        PostgresServiceServer::new(PostgresGrpcService::new(pool, providers_without_review))
            .max_decoding_message_size(MAX_PAGE_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_PAGE_MESSAGE_BYTES)
            .accept_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Gzip);

    tracing::event!(
        name: "postgres_service.server.starting",
        Level::INFO,
        server.address = %listen_addr,
        "starting postgres_service gRPC server on {{server.address}}",
    );

    Server::builder()
        .add_service(health_service)
        .add_service(grpc_service)
        .serve_with_shutdown(
            listen_addr,
            wait_for_shutdown(interrupt_signal, terminate_signal),
        )
        .await?;

    Ok(())
}

/// Waits until the process receives `SIGINT` or `SIGTERM`.
async fn wait_for_shutdown(mut interrupt_signal: Signal, mut terminate_signal: Signal) {
    tokio::select! {
        _ = interrupt_signal.recv() => {},
        _ = terminate_signal.recv() => {},
    }
    tracing::event!(
        name: "postgres_service.server.stopping",
        Level::INFO,
        "received shutdown signal, stopping postgres_service",
    );
}
