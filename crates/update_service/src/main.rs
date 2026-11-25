//! Entry point for the Echora Update Service Temporal worker process.

// Rust guideline compliant 2026-10-09

use common::{CommonSettings, init_telemetry};
use mimalloc::MiMalloc;
use tokio::signal::unix::{Signal, SignalKind, signal};
use update_service::{UpdateServiceSettings, run_workers};

/// Uses mimalloc for worker task and payload allocations.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Connects to Temporal, starts one worker per task queue, and runs until shutdown.
///
/// # Errors
///
/// Returns an error when configuration, telemetry, Temporal connection, worker
/// initialization, or worker execution fails.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let common_settings = CommonSettings::from_env()?;
    let _telemetry_guard = init_telemetry("update-service", &common_settings)?;
    let settings = UpdateServiceSettings::from_env();
    let interrupt_signal = signal(SignalKind::interrupt())?;
    let terminate_signal = signal(SignalKind::terminate())?;

    run_workers(
        &settings,
        wait_for_shutdown(interrupt_signal, terminate_signal),
    )
    .await
}

/// Waits until the process receives `SIGINT` or `SIGTERM`.
async fn wait_for_shutdown(mut interrupt_signal: Signal, mut terminate_signal: Signal) {
    tokio::select! {
        _ = interrupt_signal.recv() => {}
        _ = terminate_signal.recv() => {}
    }
}
