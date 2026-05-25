"""Populate and report observability signals for the Echora vector service.

Exercises every instrumented code path across all three layers:
  Layer 1 — AioServerInterceptor (RPC metrics, SERVER spans, error logs)
  Layer 2 — Route handler (span events, search-quality metrics)
  Layer 3 — Library processors / Qdrant client (embedding + DB metrics)

After generating traffic, queries Prometheus and prints a metric evidence
report. Optionally writes a markdown report for attaching to PRs.

Usage:
    # Basic run against local dev stack:
    PYTHONPATH=apps/vector_service/src:libs/common/src \\
        uv run python scripts/populate_vector_observability.py

    # With 6-min stress phase to trigger alert rule windows:
    uv run python scripts/populate_vector_observability.py --stress

    # With DB chaos (docker stop/start Qdrant):
    uv run python scripts/populate_vector_observability.py --chaos

    # Full comprehensive run:
    uv run python scripts/populate_vector_observability.py \\
        --stress --chaos --report docs/obs_evidence.md
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import grpc
from google.protobuf import struct_pb2
from vector_proto.v1 import vector_admin_pb2, vector_admin_pb2_grpc
from vector_proto.v1 import vector_search_pb2, vector_search_pb2_grpc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png() -> bytes:
    """Return a minimal 1×1 white RGB PNG (no external deps)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")  # filter=None, RGB white pixel
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


_MINIMAL_PNG = _make_png()

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RpcResult:
    """Outcome of a single RPC call."""

    label: str
    ok: bool
    detail: str = ""


@dataclass
class TrafficReport:
    """Aggregated results from the traffic generation phase."""

    results: list[RpcResult] = field(default_factory=list)

    def record(self, label: str, ok: bool, detail: str = "") -> None:
        self.results.append(RpcResult(label, ok, detail))
        icon = "✓" if ok else "✗"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {icon} {label}{suffix}")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate and report observability signals for the vector service."
    )
    p.add_argument("--service", default="localhost:8001",
                   help="gRPC service address (default: localhost:8001)")
    p.add_argument("--prometheus", default="http://localhost:9090",
                   help="Prometheus base URL (default: http://localhost:9090)")
    p.add_argument("--report", default="",
                   help="Write markdown evidence report to this path (optional)")
    p.add_argument("--burst", type=int, default=10,
                   help="Extra search queries for histogram distribution (default: 10)")
    p.add_argument("--wait", type=int, default=20,
                   help="Seconds to wait for OTel SDK metric export (default: 20)")
    p.add_argument("--stress", action="store_true",
                   help="Run sustained stress phase to trigger alert rule windows")
    p.add_argument("--stress-duration", type=int, default=360,
                   help="Stress phase duration in seconds (default: 360 = 6 min)")
    p.add_argument("--chaos", action="store_true",
                   help="Run DB chaos phase via docker stop/start")
    p.add_argument("--qdrant-container", default="echora-qdrant",
                   help="Qdrant Docker container name for chaos (default: echora-qdrant)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 — gRPC traffic generation
# ---------------------------------------------------------------------------

_BURST_QUERIES: list[tuple[str, str, int]] = [
    ("action adventure shonen", "anime", 10),
    ("romance school drama", "anime", 5),
    ("fantasy magic isekai", "anime", 20),
    ("science fiction space opera", "", 10),
    ("horror psychological thriller", "anime", 3),
    ("sports competition teamwork", "anime", 10),
    ("slice of life comedy", "", 15),
    ("mecha robot giant", "anime", 10),
    ("historical period drama", "", 5),
    ("music idol singer", "anime", 10),
]


async def _generate_traffic(host: str, burst: int) -> TrafficReport:
    """Send RPCs covering every instrumented code path and error case."""
    report = TrafficReport()

    async with grpc.aio.insecure_channel(host) as channel:
        adm = vector_admin_pb2_grpc.VectorAdminServiceStub(channel)
        svc = vector_search_pb2_grpc.VectorSearchServiceStub(channel)
        SR = vector_search_pb2.SearchRequest

        # Closure helpers eliminate per-call try/except boilerplate.
        async def _ok(label: str, **kw) -> None:  # success path
            try:
                r = await svc.Search(SR(**kw))
                ok = not r.HasField("error")
                report.record(label, ok, f"{len(r.data)} results" if ok else r.error.code)
            except Exception as exc:
                report.record(label, False, str(exc))

        async def _err(label: str, code: str, **kw) -> None:  # expected-error path
            try:
                r = await svc.Search(SR(**kw))
                got = r.error.code if r.HasField("error") else "no error"
                report.record(label, got == code, got)
            except Exception as exc:
                report.record(label, False, str(exc))

        # ── Admin (Health + GetStats) ──────────────────────────────────────
        print("\n  [Admin RPCs]")
        try:
            r = await adm.Health(vector_admin_pb2.HealthRequest())
            report.record("Health", r.healthy, f"healthy={r.healthy}")
        except Exception as exc:
            report.record("Health", False, str(exc))

        try:
            r = await adm.GetStats(vector_admin_pb2.GetStatsRequest())
            ok = not r.HasField("error")
            report.record("GetStats", ok, "stats received" if ok else r.error.code)
        except Exception as exc:
            report.record("GetStats", False, str(exc))

        # ── Text search success paths ──────────────────────────────────────
        print("\n  [Search — text success paths]")
        await _ok("text (no filter)", query_text="one piece pirate adventure")
        await _ok("text + entity_type=anime", query_text="romance drama", entity_type="anime", limit=5)
        # entity_type outside known set → normalised to "unknown" metric label
        await _ok("text + entity_type=movie (→unknown label)", query_text="mystery", entity_type="movie", limit=5)
        await _ok("text + limit=999 (clamped to 100)", query_text="ninja samurai", limit=999)
        # Obscure query → likely zero results → SEARCH_EMPTY_RESULTS counter
        await _ok("text + obscure query (→empty results)", query_text="xyzzy_nonexistent_title_zzz_000", limit=1)

        filt = struct_pb2.Struct()
        filt.update({"type": "TV"})
        await _ok("text + filters payload", query_text="comedy", filters=filt, limit=10)

        # ── Image paths ────────────────────────────────────────────────────
        print("\n  [Search — image paths]")
        # Valid PNG → exercises image embedding path + IMAGE_EMBEDDING_DURATION metric
        await _ok("image only (valid 1×1 PNG)", image=_MINIMAL_PNG)
        # Multimodal: text + image → dual-embedding code path
        await _ok("multimodal text+image", query_text="action adventure", image=_MINIMAL_PNG)
        # Corrupt bytes → encode_image catches exception → returns None → IMAGE_EMBEDDING_FAILED
        await _err("corrupt image (→IMAGE_EMBEDDING_FAILED)", "IMAGE_EMBEDDING_FAILED",
                   image=b"\x00\x01\x02\x03" * 100)

        # ── Error paths ────────────────────────────────────────────────────
        print("\n  [Search — error paths → RPC_ERRORS + Loki error log]")
        await _err("empty request (→MISSING_QUERY_INPUT)", "MISSING_QUERY_INPUT")
        await _err("whitespace query (→MISSING_QUERY_INPUT)", "MISSING_QUERY_INPUT", query_text="   ")

        bad = struct_pb2.Struct()
        bad.update({"genre": {"nested": {"deeply": "invalid"}}})
        await _err("bad filters (→INVALID_FILTERS)", "INVALID_FILTERS", query_text="test", filters=bad)

        # Repeat error calls so rpc_errors_total has multiple observations
        for _ in range(4):
            await svc.Search(SR())  # MISSING_QUERY_INPUT

        # ── Histogram distribution burst ───────────────────────────────────
        print(f"\n  [Burst — {burst} queries for histogram distribution]")
        burst_ok = 0
        queries = (_BURST_QUERIES * ((burst // len(_BURST_QUERIES)) + 1))[:burst]
        for query_text, entity_type, limit in queries:
            try:
                kw = {"query_text": query_text, "limit": limit}
                if entity_type:
                    kw["entity_type"] = entity_type
                await svc.Search(SR(**kw))
                burst_ok += 1
            except Exception:
                pass
        report.record(f"Burst searches ({burst_ok}/{burst})", burst_ok > 0, "histogram buckets populated")

        for _ in range(3):
            try:
                await adm.Health(vector_admin_pb2.HealthRequest())
            except Exception:
                pass

    return report


# ---------------------------------------------------------------------------
# Phase Stress — sustained traffic to cross alert rule windows
# ---------------------------------------------------------------------------


async def _run_stress_phase(host: str, duration_secs: int) -> None:
    """Sustain 10% error rate and 30% empty-result rate until alert windows fire.

    Alert targets:
      EchoraRPCHighErrorRate  — >5%   error rate for 5 min
      EchoraHighEmptyResultRate — >10% empty rate for 10 min
    Each cycle: 1 error + 3 empty + 6 valid queries. Every 5th cycle adds a
    20-concurrent burst to spike P99 above the 500 ms threshold.
    """
    print(f"\n[Phase Stress] Sustaining traffic for {duration_secs}s (~{duration_secs // 60}m)")
    print("  Targets: EchoraRPCHighErrorRate >5% (5m), EchoraHighEmptyResultRate >10% (10m)")
    end = time.time() + duration_secs
    cycle = 0

    async with grpc.aio.insecure_channel(host) as channel:
        svc = vector_search_pb2_grpc.VectorSearchServiceStub(channel)
        SR = vector_search_pb2.SearchRequest

        while time.time() < end:
            cycle += 1
            remaining = max(0, int(end - time.time()))
            print(f"\r  Cycle {cycle} | {remaining}s remaining  ", end="", flush=True)

            await svc.Search(SR())  # 1 MISSING_QUERY_INPUT → 10% error rate

            for _ in range(3):  # 3 empty-result queries → 30% empty rate
                await svc.Search(SR(query_text="xyzzy_nonexistent_stress_999", limit=1))

            for q in ["action", "romance", "fantasy", "sci-fi", "horror", "sports"]:
                await svc.Search(SR(query_text=q))

            if cycle % 5 == 0:  # P99 spike via concurrent burst
                await asyncio.gather(
                    *[svc.Search(SR(query_text="concurrent stress load", limit=50)) for _ in range(20)],
                    return_exceptions=True,
                )

            await asyncio.sleep(1)

    print(f"\n  Done ({cycle} cycles, ~{cycle * 10} requests sent)")


# ---------------------------------------------------------------------------
# Phase Chaos — DB unavailability via docker stop/start
# ---------------------------------------------------------------------------


def _run_chaos_phase(host: str, container: str) -> None:
    """Stop Qdrant, send failing requests, restart — exercises DB error metrics.

    Triggers echora_db_errors_total and the EchoraDBErrors alert rule.
    Requires the docker CLI to be available and the container name to be correct.
    """
    print(f"\n[Phase Chaos] Stopping Qdrant container '{container}'...")
    result = subprocess.run(["docker", "stop", container], capture_output=True, text=True)  # noqa: S603, S607
    if result.returncode != 0:
        print(f"  ✗ docker stop failed: {result.stderr.strip()}")
        return
    print("  ✓ Container stopped")
    print("  Sending 3 searches against downed DB (expect SEARCH_FAILED)...")

    async def _chaos_requests() -> None:
        async with grpc.aio.insecure_channel(host) as ch:
            svc = vector_search_pb2_grpc.VectorSearchServiceStub(ch)
            for i in range(3):
                try:
                    r = await svc.Search(vector_search_pb2.SearchRequest(query_text="chaos test"))
                    got = r.error.code if r.HasField("error") else "no error"
                    icon = "✓" if got == "SEARCH_FAILED" else "✗"
                    print(f"    {icon} request {i + 1}: {got}")
                except Exception as exc:
                    print(f"    ✗ request {i + 1}: {exc}")

    asyncio.run(_chaos_requests())

    print(f"  Restarting '{container}'...")
    result = subprocess.run(["docker", "start", container], capture_output=True, text=True)  # noqa: S603, S607
    if result.returncode != 0:
        print(f"  ✗ docker start failed: {result.stderr.strip()}")
        return
    print("  ✓ Restarted — waiting 15s for Qdrant health...")
    time.sleep(15)
    print("  ✓ Chaos phase complete")


# ---------------------------------------------------------------------------
# Phase 2 — Prometheus evidence
# ---------------------------------------------------------------------------


def _prom_query(base_url: str, query: str) -> tuple[str, str | None]:
    """Run a single instant PromQL query and return (value_str, error_str)."""
    params = urllib.parse.urlencode({"query": query})
    url = f"{base_url.rstrip('/')}/api/v1/query?{params}"
    try:
        req = urllib.request.Request(url=url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        return "", str(exc)

    if payload.get("status") != "success":
        return "", f"status={payload.get('status')}"

    results = payload.get("data", {}).get("result", [])
    if not results:
        return "—", None

    raw = results[0].get("value", [None, "—"])[1]
    try:
        return f"{float(raw):.4g}", None
    except (TypeError, ValueError):
        return str(raw), None


# Metric queries grouped by layer — mirrors the README signal map.
_METRIC_QUERIES: list[tuple[str, str, str]] = [
    # Layer 1 — Interceptor
    ("Layer 1 — Interceptor", "rpc_requests_total [Search]",
     'sum(echora_rpc_requests_total{rpc_method="Search"})'),
    ("", "rpc_requests_total [Health]",
     'sum(echora_rpc_requests_total{rpc_method="Health"})'),
    ("", "rpc_requests_total [GetStats]",
     'sum(echora_rpc_requests_total{rpc_method="GetStats"})'),
    ("", "rpc_errors_total [MISSING_QUERY_INPUT]",
     'sum(echora_rpc_errors_total{error_code="MISSING_QUERY_INPUT"})'),
    ("", "rpc_errors_total [INVALID_FILTERS]",
     'sum(echora_rpc_errors_total{error_code="INVALID_FILTERS"})'),
    ("", "rpc_errors_total [IMAGE_EMBEDDING_FAILED]",
     'sum(echora_rpc_errors_total{error_code="IMAGE_EMBEDDING_FAILED"})'),
    ("", "rpc_errors_total [SEARCH_FAILED]",
     'sum(echora_rpc_errors_total{error_code="SEARCH_FAILED"})'),
    ("", "error_rate % [Search]",
     'sum(rate(echora_rpc_errors_total{rpc_method="Search"}[5m])) / '
     'sum(rate(echora_rpc_requests_total{rpc_method="Search"}[5m])) * 100'),
    ("", "rpc_duration_seconds p50 [Search]",
     'histogram_quantile(0.5, sum(rate(echora_rpc_duration_seconds_bucket{rpc_method="Search"}[5m])) by (le))'),
    ("", "rpc_duration_seconds p99 [Search]",
     'histogram_quantile(0.99, sum(rate(echora_rpc_duration_seconds_bucket{rpc_method="Search"}[5m])) by (le))'),
    ("", "inflight_rpcs",
     "echora_inflight_rpcs"),
    # Layer 2 — Handler
    ("Layer 2 — Handler", "search_results_count p50",
     "histogram_quantile(0.5, sum(rate(echora_search_results_count_bucket[5m])) by (le))"),
    ("", "search_empty_results_total",
     "sum(echora_search_empty_results_total)"),
    ("", "empty_result_rate %",
     "sum(rate(echora_search_empty_results_total[5m])) / "
     'sum(rate(echora_rpc_requests_total{rpc_method="Search"}[5m])) * 100'),
    # Layer 3 — Libraries
    ("Layer 3 — Libraries", "embedding_duration_seconds p50 [text]",
     'histogram_quantile(0.5, sum(rate(echora_embedding_duration_seconds_bucket{modality="text"}[5m])) by (le))'),
    ("", "embedding_duration_seconds p99 [text]",
     'histogram_quantile(0.99, sum(rate(echora_embedding_duration_seconds_bucket{modality="text"}[5m])) by (le))'),
    ("", "embedding_duration_seconds p50 [image]",
     'histogram_quantile(0.5, sum(rate(echora_embedding_duration_seconds_bucket{modality="image"}[5m])) by (le))'),
    ("", "db_query_duration_seconds p50",
     "histogram_quantile(0.5, sum(rate(echora_db_query_duration_seconds_bucket[5m])) by (le))"),
    ("", "db_errors_total",
     "sum(echora_db_errors_total)"),
    ("", "image_download_duration_seconds p50",
     "histogram_quantile(0.5, sum(rate(echora_image_download_duration_seconds_bucket[5m])) by (le))"),
    ("", "image_download_failures_total",
     "sum(echora_image_download_failures_total)"),
    # OTel Collector health
    ("OTel Collector", "accepted_spans/s",
     "sum(rate(otelcol_receiver_accepted_spans_total[5m]))"),
    ("", "refused_spans/s",
     "sum(rate(otelcol_receiver_refused_spans_total[5m]))"),
    ("", "failed_export_spans/s (0 = healthy)",
     "sum(rate(otelcol_exporter_enqueue_failed_spans_total[5m]))"),
    ("", "batch_avg_size",
     "sum(rate(otelcol_processor_batch_batch_send_size_sum[5m])) / "
     "sum(rate(otelcol_processor_batch_batch_send_size_count[5m]))"),
]


def _collect_metrics(prometheus_url: str) -> list[tuple[str, str, str, str | None]]:
    """Query all metrics and return rows for display."""
    return [
        (section, name, *_prom_query(prometheus_url, query))
        for section, name, query in _METRIC_QUERIES
    ]


def _print_metrics(rows: list[tuple[str, str, str, str | None]]) -> None:
    """Print metric rows to stdout with section headers."""
    current_section = ""
    for section, name, value, error in rows:
        if section and section != current_section:
            print(f"\n  [{section}]")
            current_section = section
        if error and value == "":
            print(f"  {'✗':2} {name:<55} ERROR: {error}")
        else:
            display = value if not error else f"{value} (warn: {error})"
            print(f"  {'·':2} {name:<55} {display}")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_report(
    *,
    traffic: TrafficReport,
    metric_rows: list[tuple[str, str, str, str | None]],
    prometheus_url: str,
    generated_at: str,
) -> str:
    """Render a markdown evidence report suitable for PR attachments."""
    lines = [
        "# Vector Service — Observability Evidence",
        "",
        f"Generated: `{generated_at}`  ",
        f"Prometheus: `{prometheus_url}`",
        "",
        "## Traffic Generated",
        "",
        f"Total RPC calls: **{traffic.total}** &nbsp;|&nbsp; Unreachable: **{traffic.failed}**",
        "",
        "| RPC | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for r in traffic.results:
        lines.append(f"| `{r.label}` | {'✓' if r.ok else '✗'} | {r.detail} |")

    lines += ["", "## Metric Snapshot", "", "| Layer | Metric | Value |", "| --- | --- | --- |"]
    current_section = ""
    for section, name, value, error in metric_rows:
        display_section = section if section != current_section else ""
        if section:
            current_section = section
        display_value = value if not error else f"{value} ⚠ {error}"
        lines.append(f"| {display_section} | `{name}` | `{display_value}` |")

    lines += [
        "",
        "## Notes",
        "",
        "- `failed_export_spans/s = 0` is the healthy state.",
        "- Percentile metrics (`—`) appear when rate() has too few observations; re-run after more traffic.",
        "- Chaos-phase metrics (`db_errors_total`, `SEARCH_FAILED`) only populate when `--chaos` is used.",
        "- Alert thresholds only fire after sustained traffic exceeds their `for:` windows.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run traffic generation and metric evidence collection."""
    args = _parse_args()
    div = "─" * 60
    print(div)
    print(" ECHORA VECTOR SERVICE — OBSERVABILITY POPULATION")
    print(div)

    print(f"\n[Phase 1] Generating gRPC traffic → {args.service}")
    traffic = asyncio.run(_generate_traffic(args.service, args.burst))
    print(f"\n  Total: {traffic.total} RPCs | Unreachable: {traffic.failed}")

    if args.stress:
        asyncio.run(_run_stress_phase(args.service, args.stress_duration))
        print(f"\n[Waiting {args.wait}s for stress metrics to export...]")
        time.sleep(args.wait)

    if args.chaos:
        _run_chaos_phase(args.service, args.qdrant_container)
        print(f"\n[Waiting {args.wait}s for chaos metrics to export...]")
        time.sleep(args.wait)

    # OTel SDK exports every 15 s; Prometheus scrapes every 15 s.
    # 20 s covers one full export+scrape cycle in the common case.
    print(f"\n[Waiting {args.wait}s for metric export + Prometheus scrape...]")
    time.sleep(args.wait)

    print(f"\n[Phase 2] Collecting Prometheus evidence → {args.prometheus}")
    metric_rows = _collect_metrics(args.prometheus)
    _print_metrics(metric_rows)

    generated_at = dt.datetime.now(dt.UTC).isoformat()
    if args.report:
        report_str = _render_report(
            traffic=traffic,
            metric_rows=metric_rows,
            prometheus_url=args.prometheus,
            generated_at=generated_at,
        )
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_str, encoding="utf-8")
        print(f"\n[Report] Written to {out}")

    print(f"\n{div}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
