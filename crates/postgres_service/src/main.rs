//! Entry point for the Echora PostgreSQL gRPC service.

// Rust guideline compliant 2026-10-09

use common::{CommonSettings, init_telemetry};
use mimalloc::MiMalloc;
use tonic::transport::Server;
use tracing::Level;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let settings = CommonSettings::from_env()?;
    let _telemetry_guard = init_telemetry("postgres-service", &settings)?;
    let listen_addr = settings.postgres_service_addr()?;

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<tonic_health::pb::health_server::HealthServer<tonic_health::server::HealthReporter>>()
        .await;

    tracing::event!(
        name: "postgres_service.server.starting",
        Level::INFO,
        server.address = %listen_addr,
        "starting postgres_service gRPC server on {{server.address}}",
    );

    Server::builder()
        .add_service(health_service)
        .serve_with_shutdown(listen_addr, wait_for_shutdown())
        .await?;

    Ok(())
}

async fn wait_for_shutdown() {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            tracing::event!(
                name: "postgres_service.signal.error",
                Level::ERROR,
                error.message = %err,
                "failed to listen for ctrl-c signal: {{error.message}}",
            );
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(err) => {
                tracing::event!(
                    name: "postgres_service.signal.error",
                    Level::ERROR,
                    error.message = %err,
                    "failed to install SIGTERM handler: {{error.message}}",
                );
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    tracing::event!(
        name: "postgres_service.server.stopping",
        Level::INFO,
        "received shutdown signal, stopping postgres_service",
    );
}
