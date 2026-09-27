# Vector service benchmarks

Tools that measure how many searches one vector service instance handles, at
what latency and with what search accuracy, and that find where the time
goes. Findings are recorded in Linear ECHO-54; the plan and the recommended
settings are in `docs/vector_service_capacity_plan.md`.

| Folder | Contents |
| -- | -- |
| `load/` | k6 load test (`vector_search.js`, `run_vector_search.sh`), its queries, and `results/` (git-ignored) |
| `test_data/` | Builds the test collections and the query file |
| `quality/` | Search accuracy against exact search, and Qdrant's cost per search |
| `diagnostics/` | Profiling and per-stage timing of a running service |
| `reports/` | Turns a load test's raw samples into per-window numbers |

The tests of the service's own code stay in `tests/`; these tools measure it.

## Load test

A k6 test for `VectorSearchService/Search`. Every test type starts searches on
a schedule (arrival rate), whether or not earlier searches have answered. A
test that waits for answers before sending more would slow down with the
service and hide its slowdown.

### Running

Needs Docker; k6 runs from the pinned `grafana/k6` image.

```bash
benchmarks/vector_service/load/run_vector_search.sh smoke
benchmarks/vector_service/load/run_vector_search.sh load -e RATE=40 -e HOLD=5m
benchmarks/vector_service/load/run_vector_search.sh breakpoint -e RATE=4 -e MAX_RATE=40 -e BREAKPOINT_DURATION=10m
```

Against a service running outside Docker, point at it and name its process:

```bash
SERVICE_PID=$(pgrep -f vector_service.main) \
  benchmarks/vector_service/load/run_vector_search.sh breakpoint -e TARGET=localhost:8001 -e MAX_RATE=200
```

### Running against cloud infrastructure

The same script runs against a deployed service:

```bash
SAMPLE_RESOURCES=false K6_PROMETHEUS_URL=https://<prometheus>/api/v1/write \
  benchmarks/vector_service/load/run_vector_search.sh load -e TARGET=<host>:443 -e TLS=true -e RATE=200
```

- Run k6 on a machine in the same region as the service, so network distance
  is not counted as service latency.
- `SAMPLE_RESOURCES=false` skips the local CPU/GPU sampling; read the
  service's resources from its own monitoring.
- One k6 instance is enough to test one service instance. For the full
  10,000–20,000 searches/s target, split the load across several k6 machines
  (for example with the k6 Kubernetes operator) and keep each machine's CPU
  under ~70%.
- Turn off autoscaling on the service while running a breakpoint test, or the
  test finds the account's limit instead of the instance's.

### Test types

Run them in this order; each assumes the one before passed.

| Type | Load | Length | Answers |
| -- | -- | -- | -- |
| `smoke` | 1 search/s | 30 s | Does the test work and the service answer? |
| `load` | `RATE` | 2 min ramp, `HOLD` (10 min), 1 min down | Does one instance hold its target rate? |
| `stress` | `STRESS_RATE` (RATE × 1.5) | 5 min ramp, `HOLD`, 2 min down | How far does latency degrade above the target? |
| `spike` | `SPIKE_RATE` (RATE × 3) | 30 s jump, 1 min hold, 30 s drop, 2 min recovery | Does it survive a sudden rush and recover? |
| `soak` | `RATE` | `SOAK_HOLD` (3 h) | Does latency or memory creep up over time? |
| `breakpoint` | ramps to `MAX_RATE` (RATE × 10) | `BREAKPOINT_DURATION` (20 min) | Where is the limit? Stops once p95 passes `ABORT_P95_MS` (2 s) or more than 5% of searches fail |

All settings are listed at the top of `vector_search.js`. The most used:

| Setting | Default | Meaning |
| -- | -- | -- |
| `TARGET` | `localhost:8001` | Service address |
| `RATE` | 20 | Searches/s one instance should sustain |
| `P95_MS` | not set | Fail the run when p95 exceeds this |
| `QUERY_MIX` | `40,40,20` | Weights of short, title and long queries |
| `UNIQUE` | `false` | Append a counter to every query, so no cache can answer it |
| `WITH_PAYLOAD` | `true` | `false` asks for IDs and scores only |

### Queries

`search_queries.json` holds real queries in three kinds, since query length
drives embedding cost: short tag phrases, titles, and synopsis sentences.
Rebuild it with `uv run python benchmarks/vector_service/test_data/build_load_test_queries.py`.

### Results

- Summary per run in `benchmarks/vector_service/load/results/<run_id>.json`, with latency per query kind.
- CPU and memory of the service and of k6, and GPU use, in
  `benchmarks/vector_service/load/results/<run_id>-resources.csv`. If k6's CPU is near its
  limit, k6 is the bottleneck, not the service.
- Every sample in `benchmarks/vector_service/load/results/<run_id>-samples.csv.gz`. For completed
  searches and latency per 20 s window (the way to read a breakpoint test):

  ```bash
  uv run python benchmarks/vector_service/reports/summarize_load_test.py <run_id>
  ```

- Live in Prometheus as `k6_*` metrics, tagged with `run_id` and `test_type`,
  next to the service's own metrics. The latency percentiles there
  (`k6_grpc_req_duration_p95` and the others) cover every search since the run
  started, not the last few seconds, so late in a ramp they read low. Use the
  summary script for latency at a given rate.

## Profiling the service

`profile_service.sh` holds a steady rate with the `load` test and, once the
ramp is over, records 25 s of py-spy samples from the service process:

```bash
SERVICE_PID=<pid> benchmarks/vector_service/diagnostics/profile_service.sh 150 localhost:8001
```

The service must run outside Docker so py-spy can attach to it. Options after
the target go to k6 (for example `-e WITH_PAYLOAD=false`). The profile is
written as folded stacks to `benchmarks/vector_service/load/results/profile-<rate>-<time>.txt`;
open it in speedscope.app, or summarize it by thread and area of the code:

```bash
uv run python benchmarks/vector_service/diagnostics/summarize_profile.py benchmarks/vector_service/load/results/profile-<rate>-<time>.txt
```

py-spy's non-blocking mode drops samples under load, so its busy
percentages read low; the split between areas is still usable. For exact CPU
per thread, read `/proc/<pid>/task/*/stat` during the run.

## Where searches wait

`benchmarks/vector_service/diagnostics/run_timed_vector_service.py` runs the service with timing around
each stage and prints, every 10 s, how long searches wait for the model and
for Qdrant, each batch call's duration and size, and event loop lag. Run it
outside Docker with the service's usual environment variables, then load it
with the `load` test at a steady rate. It wraps the service's own functions,
so it is for diagnosis only.

## Search accuracy and Qdrant cost

Changes to Qdrant's search settings can change results, so each one is
measured for accuracy as well as speed with `benchmarks/vector_service/quality/measure_qdrant_search.py`:

```bash
./pants run benchmarks/vector_service/quality/measure_qdrant_search.py -- queries
./pants run benchmarks/vector_service/quality/measure_qdrant_search.py -- accuracy today rescore ef512_rescore
./pants run benchmarks/vector_service/quality/measure_qdrant_search.py -- cost today rescore --collection anime_load_test
```

- `queries` embeds a fixed sample of `search_queries.json` (327 queries) once,
  into `data/search_quality/queries.json`. It needs the text model; the other
  steps do not.
- `accuracy` scores each setting against an exact search with ranx: recall@10
  and NDCG@10, for the dense vector alone and for the service's hybrid query,
  overall and per query kind. Run it against a collection of real points
  (default `anime_accuracy_test`); the noisy copies in `anime_load_test` lower
  recall on their own.
- `cost` runs the hybrid query in batches of 32, 4 in flight, and reports
  Qdrant's time per query, queries/s and, for a local Qdrant container, its CPU
  per query.
- The settings are named in `SEARCH_SETTINGS` at the top of the script. Qdrant's
  address and API key come from `QDRANT_URL` and `QDRANT_API_KEY`.
- `--candidates` sets how many candidates each hybrid branch fetches (100, as
  the service does).

The embedding model's output is checked against FlagEmbedding's own `encode`
by `tests/libs/vector_processing/integration/test_flagembedding_parity.py`
(`./pants test tests/libs/vector_processing/integration::`), since the service
runs the model itself in one pass.
