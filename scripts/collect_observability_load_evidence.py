"""Collect observability load-test evidence from Prometheus and write a report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

METRIC_QUERIES: dict[str, str] = {
    "accepted_spans_rate": "sum(rate(otelcol_receiver_accepted_spans[5m]))",
    "failed_spans_rate": "sum(rate(otelcol_exporter_send_failed_spans[5m]))",
    "refused_spans_rate": "sum(rate(otelcol_processor_memory_limiter_refused_spans[5m]))",
    "batch_send_size": "avg(otelcol_processor_batch_batch_send_size)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Prometheus evidence for observability load-test tuning."
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus base URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--output",
        default="docs/observability_load_test_evidence.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any metric query fails",
    )
    return parser.parse_args()


def query_prometheus(prometheus_url: str, query: str) -> tuple[str | None, str | None]:
    parsed = urllib.parse.urlparse(prometheus_url)
    if parsed.scheme not in {"http", "https"}:
        return None, f"unsupported scheme '{parsed.scheme}'"

    params = urllib.parse.urlencode({"query": query})
    url = f"{prometheus_url.rstrip('/')}/api/v1/query?{params}"
    request = urllib.request.Request(url=url, method="GET")  # noqa: S310

    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return None, str(exc)

    if payload.get("status") != "success":
        return None, f"status={payload.get('status')}"

    results = payload.get("data", {}).get("result", [])
    if not results:
        return None, "no data"

    # Prometheus vector result value shape: [timestamp, "value"]
    value = results[0].get("value", [None, None])[1]
    return value, None


def render_report(
    *,
    prometheus_url: str,
    results: dict[str, str | None],
    errors: dict[str, str],
) -> str:
    generated_at = dt.datetime.now(dt.UTC).isoformat()

    lines = [
        "# Observability Load Test Evidence",
        "",
        f"Generated at: `{generated_at}`",
        f"Prometheus source: `{prometheus_url}`",
        "",
        "## Metric Snapshot",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]

    for metric_name, value in results.items():
        rendered_value = value if value is not None else "N/A"
        lines.append(f"| `{metric_name}` | `{rendered_value}` |")

    lines.extend(["", "## Query Errors", ""])
    if errors:
        for metric_name, error in errors.items():
            lines.append(f"- `{metric_name}`: {error}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Attach this report to tuning PRs for collector memory-limiter, batch, and sampling changes.",
            "- Compare failed/refused span rates before and after configuration changes.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    results: dict[str, str | None] = {}
    errors: dict[str, str] = {}

    for metric_name, query in METRIC_QUERIES.items():
        value, error = query_prometheus(args.prometheus_url, query)
        results[metric_name] = value
        if error is not None:
            errors[metric_name] = error

    report = render_report(
        prometheus_url=args.prometheus_url,
        results=results,
        errors=errors,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
