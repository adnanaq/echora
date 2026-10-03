#!/usr/bin/env bash
# Run the vector search load test in the pinned k6 image.
#
# Usage: benchmarks/vector_service/load/run_vector_search.sh TEST_TYPE [extra k6 args...]
#   benchmarks/vector_service/load/run_vector_search.sh smoke
#   benchmarks/vector_service/load/run_vector_search.sh load -e RATE=40 -e HOLD=5m
#   benchmarks/vector_service/load/run_vector_search.sh breakpoint -e RATE=20 -e MAX_RATE=400
#
# A wrapper around run_load_test.py (see there for all options). Results go to
# benchmarks/vector_service/load/results/ and to Prometheus (K6_PROMETHEUS_URL,
# default the local observability stack). The service's resources are read
# from SERVICE_PID (a service run outside Docker) or SERVICE_CONTAINER (default
# echora-dev-vector-service); SAMPLE_RESOURCES=false skips sampling, for a
# service that runs elsewhere (cloud).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
exec uv run --quiet python -m benchmarks.vector_service.load.run_load_test "$@"
