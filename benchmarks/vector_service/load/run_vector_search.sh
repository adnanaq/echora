#!/usr/bin/env bash
# Run the vector search load test in the pinned k6 image.
#
# Usage: benchmarks/vector_service/load/run_vector_search.sh TEST_TYPE [extra k6 args...]
#   benchmarks/vector_service/load/run_vector_search.sh smoke
#   benchmarks/vector_service/load/run_vector_search.sh load -e RATE=40 -e HOLD=5m
#   benchmarks/vector_service/load/run_vector_search.sh breakpoint -e RATE=20 -e MAX_RATE=400
#
# Results are pushed to Prometheus (K6_PROMETHEUS_URL, default the local
# observability stack). benchmarks/vector_service/load/results/ gets, per run:
#   <run_id>.json            end-of-run summary
#   <run_id>-resources.csv   CPU and memory of the service and of k6, plus GPU
#                            use, sampled every 2 s
# The service is found by SERVICE_CONTAINER (default echora-dev-vector-service)
# or, for a service run outside Docker, by SERVICE_PID. Set SAMPLE_RESOURCES=false
# when the service runs elsewhere (cloud); read its resources from its own
# monitoring instead.
set -euo pipefail

K6_IMAGE="grafana/k6:2.3.0"
SAMPLE_SECONDS=2
TEST_TYPE="${1:?usage: $0 TEST_TYPE [extra k6 args...]}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/vector_service/load/results"
RUN_ID="${TEST_TYPE}-$(date +%Y%m%d-%H%M%S)"
K6_CONTAINER="k6-${RUN_ID}"
SERVICE_CONTAINER="${SERVICE_CONTAINER:-echora-dev-vector-service}"
mkdir -p "${RESULTS_DIR}"

PREVIOUS_CPU_TICKS=""
PREVIOUS_SAMPLE_NS=""
SERVICE_SAMPLE=","

# Sets SERVICE_SAMPLE to "cpu,memory". For a process, CPU is measured from its
# CPU ticks in /proc since the previous sample; ps %cpu would give the average
# since the process started.
sample_service() {
  if [[ -z "${SERVICE_PID:-}" ]]; then
    SERVICE_SAMPLE="$(docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' \
      "${SERVICE_CONTAINER}" 2>/dev/null | sed 's| / .*||')"
    return
  fi
  local ticks now_ns cpu="" memory
  ticks="$(awk '{print $14 + $15}' "/proc/${SERVICE_PID}/stat")"
  now_ns="$(date +%s%N)"
  if [[ -n "${PREVIOUS_CPU_TICKS}" ]]; then
    cpu="$(awk -v ticks="$((ticks - PREVIOUS_CPU_TICKS))" -v ns="$((now_ns - PREVIOUS_SAMPLE_NS))" \
      -v hz="$(getconf CLK_TCK)" 'BEGIN {printf "%.1f%%", 100 * ticks / hz / (ns / 1e9)}')"
  fi
  PREVIOUS_CPU_TICKS="${ticks}"
  PREVIOUS_SAMPLE_NS="${now_ns}"
  memory="$(ps -p "${SERVICE_PID}" -o rss= | awk '{printf "%.0fMiB", $1 / 1024}')"
  SERVICE_SAMPLE="${cpu},${memory}"
}

sample_resources() {
  local csv="${RESULTS_DIR}/${RUN_ID}-resources.csv"
  echo "time,service_cpu,service_memory,k6_cpu,gpu_util,gpu_memory_mib" >"${csv}"
  while true; do
    local k6_cpu gpu
    sample_service
    k6_cpu="$(docker stats --no-stream --format '{{.CPUPerc}}' "${K6_CONTAINER}" 2>/dev/null || true)"
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null |
      head -1 | tr -d ' ' || true)"
    echo "$(date +%H:%M:%S),${SERVICE_SAMPLE},${k6_cpu},${gpu:-,}" >>"${csv}"
    sleep "${SAMPLE_SECONDS}"
  done
}

if [[ "${SAMPLE_RESOURCES:-true}" == "true" ]]; then
  sample_resources &
  SAMPLER_PID=$!
  trap 'kill "${SAMPLER_PID}" 2>/dev/null || true' EXIT
fi

docker run --rm --name "${K6_CONTAINER}" --network host \
  --user "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:/repo" \
  -w /repo/benchmarks/vector_service/load \
  -e K6_PROMETHEUS_RW_SERVER_URL="${K6_PROMETHEUS_URL:-http://localhost:9090/api/v1/write}" \
  -e K6_PROMETHEUS_RW_TREND_STATS="p(50),p(95),p(99),max" \
  "${K6_IMAGE}" run \
  --out experimental-prometheus-rw \
  --out "csv=results/${RUN_ID}-samples.csv.gz" \
  --tag run_id="${RUN_ID}" \
  --new-machine-readable-summary \
  --summary-export "results/${RUN_ID}.json" \
  -e TEST_TYPE="${TEST_TYPE}" \
  "$@" \
  vector_search.js
