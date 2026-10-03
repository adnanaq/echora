from datetime import datetime
from pathlib import Path

from benchmarks.vector_service.toolkit.k6_runner import (
    LoadRun,
    k6_command,
    new_run_id,
)
from benchmarks.vector_service.toolkit.settings import REPOSITORY_ROOT, load_environment


def test_new_run_id_returns_type_and_timestamp_format() -> None:
    assert new_run_id("breakpoint", datetime(2026, 9, 27, 13, 59, 9)) == (
        "breakpoint-20260927-135909"
    )


def test_k6_command_default_settings_matches_shell_script() -> None:
    environment = load_environment("laptop", ["k6.html_report=false"])

    command = k6_command(
        environment,
        run_id="load-20260927-120000",
        test_type="load",
        k6_arguments=["-e", "RATE=40"],
        user="1000:1000",
    )

    assert command == [
        "docker", "run", "--rm", "--name", "k6-load-20260927-120000",
        "--network", "host", "--user", "1000:1000",
        "-v", f"{REPOSITORY_ROOT}:/repo",
        "-w", "/repo/benchmarks/vector_service/load",
        "-e", "K6_PROMETHEUS_RW_SERVER_URL=http://localhost:9090/api/v1/write",
        "-e", "K6_PROMETHEUS_RW_TREND_STATS=p(50),p(95),p(99),max",
        "-e", "K6_CSV_TIME_FORMAT=unix_milli",
        "grafana/k6:2.3.0", "run",
        "--out", "experimental-prometheus-rw",
        "--out", "csv=results/load-20260927-120000-samples.csv.gz",
        "--tag", "run_id=load-20260927-120000",
        "--new-machine-readable-summary",
        "--summary-export", "results/load-20260927-120000.json",
        "-e", "TEST_TYPE=load",
        "-e", "RATE=40",
        "vector_search.js",
    ]  # fmt: skip


def test_k6_command_html_report_and_target_asked_adds_them() -> None:
    environment = load_environment("laptop")

    command = k6_command(
        environment,
        run_id="smoke-20260927-120000",
        test_type="smoke",
        k6_arguments=[],
        user="1000:1000",
        target="localhost:8011",
    )

    assert "K6_WEB_DASHBOARD=true" in command
    assert "K6_WEB_DASHBOARD_PORT=-1" in command
    assert (
        "K6_WEB_DASHBOARD_EXPORT=results/smoke-20260927-120000-report.html" in command
    )
    assert command[-5:] == [
        "-e", "TARGET=localhost:8011", "-e", "TLS=false", "vector_search.js",
    ]  # fmt: skip


def test_k6_command_results_outside_repository_get_own_mount(tmp_path: Path) -> None:
    environment = load_environment(
        "laptop", [f"results_dir={tmp_path}", "k6.html_report=false"]
    )

    command = k6_command(
        environment, run_id="smoke-1", test_type="smoke", k6_arguments=[], user="1:1"
    )

    assert f"{tmp_path}:/results" in command
    assert "csv=/results/smoke-1-samples.csv.gz" in command


def test_load_run_names_samples_resources_and_summary_files(tmp_path: Path) -> None:
    run = LoadRun(run_id="smoke-1", results_dir=tmp_path)

    assert run.samples_file == tmp_path / "smoke-1-samples.csv.gz"
    assert run.resources_file == tmp_path / "smoke-1-resources.csv"
    assert run.summary_file == tmp_path / "smoke-1.json"
    assert run.html_report_file == tmp_path / "smoke-1-report.html"
    assert run.console_file == tmp_path / "smoke-1-console.txt"
