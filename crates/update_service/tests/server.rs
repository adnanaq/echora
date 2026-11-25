//! Integration tests that run the `update_service` binary against Temporal.

#![allow(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

use std::process::{Child, Command, ExitStatus};
use std::thread::sleep;
use std::time::{Duration, Instant};

const WORKER_WAIT: Duration = Duration::from_secs(15);
const POLL_INTERVAL: Duration = Duration::from_millis(100);

fn start_worker_process(temporal_address: &str) -> Child {
    Command::new(env!("CARGO_BIN_EXE_update_service"))
        .env("TEMPORAL_ADDRESS", temporal_address)
        .env("OTEL_ENABLED", "false")
        .spawn()
        .expect("start update_service")
}

fn wait_until_running(worker: &mut Child) {
    sleep(Duration::from_millis(750));
    assert!(
        worker.try_wait().expect("check update_service").is_none(),
        "update_service exited before receiving shutdown signal"
    );
}

fn wait_for_exit(worker: &mut Child) -> ExitStatus {
    let deadline = Instant::now() + WORKER_WAIT;
    loop {
        if let Some(exit_status) = worker.try_wait().expect("check update_service") {
            return exit_status;
        }
        assert!(Instant::now() < deadline, "update_service did not stop");
        sleep(POLL_INTERVAL);
    }
}

fn send_signal(worker: &Child, signal_name: &str) {
    let kill_status = Command::new("kill")
        .args(["-s", signal_name, &worker.id().to_string()])
        .status()
        .expect("run kill");
    assert!(kill_status.success(), "kill -s {signal_name} failed");
}

#[test]
fn update_service_shutdown_signal_stops_workers_cleanly() {
    let temporal_address =
        std::env::var("TEMPORAL_ADDRESS").unwrap_or_else(|_| "http://localhost:7233".to_owned());
    for signal_name in ["TERM", "INT"] {
        let mut worker = start_worker_process(&temporal_address);
        wait_until_running(&mut worker);

        send_signal(&worker, signal_name);
        let exit_status = wait_for_exit(&mut worker);

        assert!(
            exit_status.success(),
            "SIG{signal_name} exit: {exit_status}"
        );
    }
}

#[test]
fn update_service_invalid_temporal_address_exits_with_error() {
    let mut worker = start_worker_process("http://127.0.0.1:1");
    let exit_status = wait_for_exit(&mut worker);
    assert!(!exit_status.success());
}
