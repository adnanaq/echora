//! Tests that run the `update_service` binary against a local Temporal dev server.

mod support;

use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail, ensure};
use support::{complete_refresh_workflow, find_free_port, start_mock_grpc_server};
use temporal_workflows::testing::start_local_environment;
use uuid::Uuid;

const PROCESS_WAIT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(100);

struct WorkerProcess {
    child: Child,
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn start_worker_process(extra_vars: &[(&str, &str)]) -> anyhow::Result<WorkerProcess> {
    let missing_config_file = Path::new(env!("CARGO_TARGET_TMPDIR")).join("missing-temporal.toml");
    let mut command = Command::new(env!("CARGO_BIN_EXE_update_service"));
    command
        .env_clear()
        .env("TEMPORAL_CONFIG_FILE", missing_config_file)
        .env("OTEL_ENABLED", "false")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Ok(profile_file) = std::env::var("LLVM_PROFILE_FILE") {
        command.env("LLVM_PROFILE_FILE", profile_file);
    }
    for (key, value) in extra_vars {
        command.env(key, value);
    }
    let child = command.spawn().context("start update_service")?;
    Ok(WorkerProcess { child })
}

async fn wait_for_exit(worker: &mut Child) -> anyhow::Result<ExitStatus> {
    let deadline = Instant::now() + PROCESS_WAIT;
    loop {
        if let Some(exit_status) = worker.try_wait()? {
            return Ok(exit_status);
        }
        if Instant::now() >= deadline {
            worker.kill()?;
            worker.wait()?;
            bail!("update_service did not exit within {PROCESS_WAIT:?}");
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn read_stderr(worker: &mut Child) -> anyhow::Result<String> {
    let mut stderr_text = String::new();
    if let Some(mut stderr) = worker.stderr.take() {
        std::io::Read::read_to_string(&mut stderr, &mut stderr_text)?;
    }
    Ok(stderr_text)
}

fn send_signal(worker: &Child, signal_name: &str) -> anyhow::Result<()> {
    let kill_status = Command::new("kill")
        .args(["-s", signal_name, &worker.id().to_string()])
        .status()
        .context("run kill")?;
    ensure!(kill_status.success(), "kill -s {signal_name} failed");
    Ok(())
}

#[tokio::test]
async fn update_service_each_shutdown_signal_drains_workers_and_exits_zero() -> anyhow::Result<()> {
    let port = find_free_port()?;
    let environment = start_local_environment(Some(port)).await?;
    let grpc_endpoint = start_mock_grpc_server().await?;
    let address = format!("127.0.0.1:{port}");
    for signal_name in ["TERM", "INT"] {
        let mut worker = start_worker_process(&[
            ("TEMPORAL_ADDRESS", address.as_str()),
            ("POSTGRES_SERVICE_ADDRESS", grpc_endpoint.as_str()),
            ("ENRICHMENT_SERVICE_ADDRESS", grpc_endpoint.as_str()),
        ])?;
        complete_refresh_workflow(environment.client(), Uuid::now_v7(), PROCESS_WAIT).await?;

        send_signal(&worker.child, signal_name)?;
        let exit_status = wait_for_exit(&mut worker.child).await?;
        let stderr_text = read_stderr(&mut worker.child)?;
        ensure!(
            exit_status.success(),
            "SIG{signal_name} exit {exit_status}: {stderr_text}"
        );
    }
    environment.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn update_service_unreachable_server_exits_nonzero() -> anyhow::Result<()> {
    let mut worker = start_worker_process(&[("TEMPORAL_ADDRESS", "127.0.0.1:1")])?;
    let exit_status = wait_for_exit(&mut worker.child).await?;
    let stderr_text = read_stderr(&mut worker.child)?;
    ensure!(!exit_status.success());
    ensure!(
        stderr_text.contains("failed to connect to Temporal"),
        "{stderr_text}"
    );
    Ok(())
}

#[tokio::test]
async fn update_service_invalid_versioning_setting_exits_nonzero() -> anyhow::Result<()> {
    let mut worker = start_worker_process(&[("WORKER_VERSIONING_ENABLED", "maybe")])?;
    let exit_status = wait_for_exit(&mut worker.child).await?;
    let stderr_text = read_stderr(&mut worker.child)?;
    ensure!(!exit_status.success());
    ensure!(
        stderr_text.contains("WORKER_VERSIONING_ENABLED"),
        "{stderr_text}"
    );
    Ok(())
}
