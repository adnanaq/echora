#!/usr/bin/env bash
# Profile the vector service with py-spy while it serves a steady search rate.
#
# Usage: SERVICE_PID=<pid> benchmarks/vector_service/diagnostics/profile_service.sh RATE [TARGET] [k6 options...]
#
# Runs the load test at RATE searches/s (default target localhost:8001) and,
# once the ramp is over (RAMP_SECONDS, default 120, the load test's ramp; then
# SETTLE_SECONDS, default 15), records PROFILE_SECONDS (default 25) of
# py-spy samples from the service process. The service must run outside Docker
# so py-spy can attach to SERVICE_PID; attaching may need sudo or
# kernel.yama.ptrace_scope=0. Options after TARGET go to k6, for example
# -e WITH_PAYLOAD=false.
#
# Writes to benchmarks/vector_service/load/results/:
#   profile-<RATE>-<time>.txt    folded stacks, open with speedscope.app or
#                                flamegraph.pl
#   load-<RATE>-<time>.txt       the load test's console output
set -euo pipefail

if [[ $# -lt 1 || -z "${SERVICE_PID:-}" ]]; then
  echo "usage: SERVICE_PID=<pid> $0 RATE [TARGET] [k6 options...]" >&2
  exit 2
fi

RATE="$1"
TARGET="${2:-localhost:8001}"
shift $(( $# >= 2 ? 2 : 1 ))
PROFILE_SECONDS="${PROFILE_SECONDS:-25}"
RAMP_SECONDS="${RAMP_SECONDS:-120}"
SETTLE_SECONDS="${SETTLE_SECONDS:-15}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/vector_service/load/results"
RUN_TIME="$(date +%Y%m%d-%H%M%S)"
PROFILE_FILE="${RESULTS_DIR}/profile-${RATE}-${RUN_TIME}.txt"
LOAD_OUTPUT="${RESULTS_DIR}/load-${RATE}-${RUN_TIME}.txt"
HOLD_SECONDS=$((SETTLE_SECONDS + PROFILE_SECONDS + 20))

mkdir -p "${RESULTS_DIR}"
SERVICE_PID="${SERVICE_PID}" "${REPO_ROOT}/benchmarks/vector_service/load/run_vector_search.sh" load \
  -e TARGET="${TARGET}" -e RATE="${RATE}" -e HOLD="${HOLD_SECONDS}s" "$@" \
  > "${LOAD_OUTPUT}" 2>&1 &
load_pid=$!

sleep $((RAMP_SECONDS + SETTLE_SECONDS))
uvx py-spy record --pid "${SERVICE_PID}" --duration "${PROFILE_SECONDS}" \
  --rate 200 --format raw --nonblocking --output "${PROFILE_FILE}"

wait "${load_pid}" || echo "load test exited with $?; see ${LOAD_OUTPUT}" >&2
grep -a -E "iterations\.\.|grpc_req_duration\.\." "${LOAD_OUTPUT}" || true
echo "profile: ${PROFILE_FILE}"
