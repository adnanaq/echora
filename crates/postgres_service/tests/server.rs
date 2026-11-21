//! Integration tests that run the `postgres_service` binary against PostgreSQL.

#![allow(
    clippy::expect_used,
    reason = "a failed setup step stops the test, as a test's own expect does"
)]

use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, ExitStatus};
use std::thread::sleep;
use std::time::{Duration, Instant};

/// Longest wait for the server to start or stop before the test fails.
///
/// Startup runs the migrations, which takes a few seconds on a fresh database.
const SERVER_WAIT: Duration = Duration::from_secs(60);

/// Pause between checks while waiting on the server.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("reserve a local port")
        .port()
}

fn start_server(port: u16, database_url: Option<&str>, host: &str) -> Child {
    let mut server_command = Command::new(env!("CARGO_BIN_EXE_postgres_service"));
    server_command
        .env("POSTGRES_SERVICE_HOST", host)
        .env("POSTGRES_SERVICE_PORT", port.to_string())
        .env("OTEL_ENABLED", "false")
        .env_remove("DATABASE_URL");
    if let Some(url) = database_url {
        server_command.env("DATABASE_URL", url);
    }
    server_command.spawn().expect("start postgres_service")
}

fn wait_until_listening(server: &mut Child, port: u16) {
    let deadline = Instant::now() + SERVER_WAIT;
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            server.try_wait().expect("check the server").is_none(),
            "postgres_service exited before listening"
        );
        assert!(Instant::now() < deadline, "postgres_service did not start");
        sleep(POLL_INTERVAL);
    }
}

fn wait_for_exit(server: &mut Child) -> ExitStatus {
    let deadline = Instant::now() + SERVER_WAIT;
    loop {
        if let Some(exit_status) = server.try_wait().expect("check the server") {
            return exit_status;
        }
        assert!(Instant::now() < deadline, "postgres_service did not stop");
        sleep(POLL_INTERVAL);
    }
}

fn send_signal(server: &Child, signal_name: &str) {
    let kill_status = Command::new("kill")
        .args(["-s", signal_name, &server.id().to_string()])
        .status()
        .expect("run kill");
    assert!(kill_status.success(), "kill -s {signal_name} failed");
}

#[test]
fn postgres_service_shutdown_signal_stops_server_cleanly() {
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL is set for tests");
    for signal_name in ["TERM", "INT"] {
        let port = find_free_port();
        let mut server = start_server(port, Some(&database_url), "127.0.0.1");
        wait_until_listening(&mut server, port);

        send_signal(&server, signal_name);
        let exit_status = wait_for_exit(&mut server);

        assert!(
            exit_status.success(),
            "SIG{signal_name} exit: {exit_status}"
        );
    }
}

#[test]
fn postgres_service_invalid_startup_settings_exit_with_error() {
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL is set for tests");
    let mut missing_database = start_server(find_free_port(), None, "127.0.0.1");
    let mut invalid_host = start_server(find_free_port(), Some(&database_url), "not-an-ip");

    let missing_database_status = wait_for_exit(&mut missing_database);
    let invalid_host_status = wait_for_exit(&mut invalid_host);

    assert!(!missing_database_status.success());
    assert!(!invalid_host_status.success());
}
