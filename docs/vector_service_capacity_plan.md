# Vector Service Capacity Plan

How many searches one vector service instance can handle, and the changes
that raise it. Findings and numbers are recorded in Linear ECHO-54; this
document holds the plan and how each step is measured.

## Goal

- Overall target: **10,000–20,000 searches/s at peak**, across all
  instances. About 10–25× MyAnimeList's estimated peak search traffic; set high
  on purpose, since scaling down is easier than scaling up.
- Per instance: the most searches/s one instance serves at the p95 latency
  goal, on CPU and on GPU, and the cost per 1,000 searches/s.
- Scope: the vector service and Qdrant (Qdrant Cloud in production). The Rust
  backend, the PostgreSQL service and the LLM agent do not exist yet.

## Current numbers

k6 breakpoint test, realistic query mix:

| Instance | Collection | Keeps up to | p50 / p95 below the limit | Limited by |
| -- | -- | -- | -- | -- |
| CPU (dev container) | 40 points | ~4.5 searches/s | 260 ms / 300 ms | the service: one embedding per search, at most 2 at once; ~9 of 24 threads used |
| GPU (RTX 4070 Laptop) | 40 points | ~48 searches/s | 21 ms / 26 ms | the same; GPU 37–41% busy |
| GPU (RTX 4070 Laptop) | 1.3M points | ~48 searches/s | 31–58 ms / 50–69 ms | the same |
| GPU, one model pass per encode | 1.3M points | ~80 searches/s | 22 ms / 25–28 ms | the service; ~1.4 CPU cores, GPU ~30% busy |
| GPU, plus batching (up to 64, no wait) | 1.3M points | ~200 searches/s | 22–27 ms / 28–78 ms up to ~100/s | the Qdrant client: 75% of the event loop's time (profiled at 150/s) |
| GPU, plus IDs and scores only | 1.3M points | ~250 searches/s | 18–22 ms / 21–46 ms up to ~100/s | the Python side; ~1.3 CPU cores, GPU 20–35% busy |
| GPU, plus Qdrant query batching | 1.3M points | ~350 searches/s | 17–19 ms / 20–50 ms up to ~80/s; p95 ~140 ms at 250/s | the Python side; ~1.3 CPU cores, GPU 25–40% busy |
| GPU, plus gRPC to Qdrant | 1.3M points | ~345 searches/s | 15–19 ms / 18–54 ms up to ~160/s; p95 ~85 ms at 240/s | the Python side; ~1.3 CPU cores, GPU 25–40% busy |
| GPU, plus skipping the Qdrant client's inference check | 1.3M points | ~595 searches/s | per window: 13 / 17 ms up to ~95/s, 57 / 77 ms at ~310/s, 84 / 110 ms at ~490/s, 113 / 190 ms at ~570/s | the Python side and, increasingly, the GPU; ~1.4 CPU cores, GPU 35–70% busy |
| GPU, plus rescoring with 4× oversampling (hybrid recall 0.917 → 0.958) | 1.3M points | ~585 searches/s | per window: 15 / 30 ms up to ~95/s, 63 / 91 ms at ~310/s, 103 / 140 ms at ~490/s, 148 / 240 ms at ~570/s | the Python side and the GPU; ~1.4 CPU cores, GPU 40–80% busy |
| GPU, plus one model call at a time and Qdrant batches of up to 8 (16 in flight) | 1.3M points | ~590 searches/s | per window: 42 / 58 ms at ~255/s, 70 / 94 ms at ~415/s, 85 / 119 ms at ~495/s, 118 / 173 ms at ~545/s | query embedding: at the limit every model call is a full batch of 64, back to back (~600 texts/s for this query mix); service ~1.3 CPU cores, Qdrant ~5 cores, machine 39% busy |

Latency in the rows above the last one is k6's Prometheus figure, which
covers every search since the run started, so at the higher rates it reads
lower than the latency at that moment. The last row is per 20 s window, from
the run's raw samples (`benchmarks/vector_service/reports/summarize_load_test.py`). The "keeps up to"
rates come from completed counts and are not affected.

Qdrant alone at 1.3M points, hybrid search with rescoring and 4×
oversampling: ~4.6 ms for one search sent on its own, ~9–12 ms of Qdrant CPU
per search, at most ~1,200 searches/s on the laptop's 24 threads (16 batches
of 8 in flight). Inside one batch, Qdrant searches the queries one after
another in each segment, so a batch's time grows with its size.

## How every step is measured

- Load test: `benchmarks/vector_service/` (see its README). The breakpoint test finds the
  limit; the load test confirms a rate holds.
- One change at a time, each compared with the numbers before it, same query
  mix, same collection.
- Record every result as a finding in ECHO-54.
- Efficiency must not cost accuracy: every change that can alter results
  (Qdrant search settings, candidates, quantization) is measured for recall
  against an exact-search ground truth as well as for speed. Settings that
  lower recall are undone or reported as a trade-off, never adopted silently.

## Steps

### 1. Realistic test collection

Built by `benchmarks/vector_service/test_data/build_load_test_collection.py` (`records`, then
`collection`).

- [x] Build records from the ~800 anime cached in Redis, from the cache only:
      non-local lookups and browser starts are blocked, retries are off, and
      AniDB uses its own `only-if-cached` request. 827 records; characters and
      episodes from AniDB XML for 391 of them
- [x] Estimate the full catalogue: ~1.1–1.3M points (40,346 anime, 514,050
      episodes, ~770k characters as indexed today)
- [x] Fill `anime_load_test` (same settings as `anime_database`, which stays
      untouched): all 40,346 anime embedded (827 from the cached records, the
      rest from offline-database fields), all 16.7k real characters and
      episodes embedded, then characters and episodes topped up to 770,000 and
      514,050 with copies of real points' vectors plus small noise (not
      re-embedded). No image vectors. Result: 1,324,089 points
- [x] Rerun the GPU breakpoint test against it, and measure Qdrant on its own

### 2. Batch query embeddings in the service

- [x] Run the model once per encode call: FlagEmbedding's `encode` runs it
      twice (a trial pass to pick a batch size). Same output, 2.0–2.4× faster
      per call, ~50 → ~80 searches/s on the GPU
- [x] Combine concurrent searches into shared model calls
      (`RequestBatcher`; `EMBED_BATCH_MAX_SIZE`, `EMBED_BATCH_MAX_WAIT_MS`,
      off by default): ~80 → ~200 searches/s on the GPU
- [x] Tune batch size and `embed_max_concurrency`: one model call at a time
      is fastest (p50 at a steady 500/s: 82 ms with 1, 102 with 2, ~155 with
      4, ~300 with 8), with the same limit; batches of up to 128 change
      nothing. Batch wait time stays 0 (not tuned: with one call at a time,
      batches already grow while the call runs)
- [x] Find where searches wait at high load (`benchmarks/vector_service/diagnostics/run_timed_vector_service.py`
      times each stage). At a steady 560/s nothing is saturated (event loop
      thread ~43%, GPU ~53%, Qdrant ~5 cores, machine 39%); searches wait in
      the model queue (~60–90 ms) and in Qdrant batches (~55–75 ms). Qdrant
      searches the queries of one batch one after another in each segment, so
      a batch of 30 takes ~60 ms although one search takes ~4.6 ms: smaller
      Qdrant batches with more in flight (8 / 16) cut p95 near the limit. At
      the limit (~590/s) the model runs full batches back to back: query
      embedding is the limit on this GPU

### 3. Qdrant query path and collection, one change at a time

- [x] Send concurrent searches in one `query_batch_points` call
      (`QDRANT_QUERY_BATCH_*` settings, off by default): ~250 → ~350 searches/s
- [x] Return IDs and scores only (`with_payload` on `SearchRequest`, payloads
      stay the default): ~200 → ~250 searches/s, p95 30–60% lower
- [x] gRPC instead of HTTP to Qdrant (`QDRANT_PREFER_GRPC`, off by default):
      same limit, p50/p95 30–40% lower
- [x] Stop the Qdrant client searching every request for text to embed
      locally (`cloud_inference=True`; we always send finished vectors): it
      walked every number of every query vector, ~1.2 ms of CPU per query and
      ~40% of the event loop. ~345 → ~595 searches/s, same results
- [ ] Tune Qdrant step by step, accuracy first. Accuracy (recall@10, NDCG@10
      with ranx, against exact search) on `anime_accuracy_test` (56k real
      points, no copies); cost on `anime_load_test` (1.3M points). Both with
      `benchmarks/vector_service/quality/measure_qdrant_search.py` (see `benchmarks/vector_service/README.md`)
  - [x] A. Accuracy baseline: dense recall@10 0.898 on 4 segments, below
        Qdrant's typical 0.95
  - [x] B. Search-time accuracy settings: on 4 segments rescoring lifts dense
        recall to 0.949 for ~3% more Qdrant CPU; higher `hnsw_ef` costs steeply
  - [x] C. Fewer segments (`default_segment_number: 2` with
        `max_segment_size: 5000000`): ~20% less Qdrant CPU at the same
        settings, but lower recall, since one segment returns fewer candidates
        for rescoring. With oversampling (rescore + 4×: hybrid 0.955 at 11.4 ms)
        it is about even with 4 segments at equal accuracy. Both test
        collections now use the merged layout
  - [x] D. Fewer candidates per branch: 50 saves ~24% CPU for 1.6 points of
        accuracy, no cheaper than other settings at equal accuracy; keep 100
  - [x] E. Index settings (`m`, `ef_construct`): skipped. `text_vector` is
        built with `m=64, ef_construct=256`, 4× and 2.5× Qdrant's defaults,
        and search-time settings alone reach 0.972 dense recall (B), so the
        index is not what limits accuracy
  - [x] F. Search settings in the service (`QDRANT_SEARCH_HNSW_EF`,
        `QDRANT_SEARCH_RESCORE`, `QDRANT_SEARCH_OVERSAMPLING`, unset by default;
        applied to the dense text vector). Chosen: rescore with 4×
        oversampling, `hnsw_ef` unset: hybrid recall 0.917 → 0.958 on real
        points, Qdrant CPU 8.8 → 11.6 ms per query (`ef` 512 with 2× oversampling: 0.001 more recall
        for 0.8 ms more). End to end on the GPU: limit ~595 → ~585 searches/s,
        p95 ~15–25% higher at the same rate; the service's results match
        Qdrant with these settings (0.987 overlap, embedding noise)
- [ ] Build a labelled query set to measure relevance, not just agreement
      with exact search (needed before tuning RRF weights)

Remaining work, in the suggested order: step 5 (query cache), step 6's
CPU-only re-measurement, step 8 (portable measurement tools), then step 6's
model server and ONNX comparisons, which need many runs on different setups.
The smaller items in step 9 fit in between. Step 6's cloud part and step 7 wait
for a cloud account and region.

### 4. Several worker processes per instance

- [ ] On hold: only needed if the Python side becomes the limit. It is not:
      at a steady 500/s the event loop thread is ~42% busy and query
      embedding is the limit (ECHO-54 finding 24)

### 5. Query-embedding cache

Query embedding is the limit, so a search answered from a cache skips the
bottleneck. Text searches never use the existing embedding cache today
(`encode_text_with_sparse` embeds every query; finding 3).

- [ ] Read how the existing cache works (Redis, keys, what it stores) and
      whether it can hold dense + sparse text embeddings
- [ ] Cache text query embeddings; check results are unchanged
- [ ] Load-test with repeated queries and with `UNIQUE=true` (no hits); measure
      the saving, and the cost when nothing hits
- [ ] Estimate the hit rate on realistic traffic (how often the same query
      text repeats)

### 6. Hardware and hosting

- [ ] Re-measure a CPU-only instance with every change so far (production
      compose runs CPU only, 2 CPUs / 4 GB; the only CPU number is the
      original ~4.5 searches/s). Includes whether `EMBED_MAX_CONCURRENCY=1`
      also suits CPU, and a batch size for CPU
- [ ] A batching model server (Hugging Face TEI, Triton, m3serve) against
      batching in the service; first check which serve BGE-M3's sparse output
- [ ] CPU inference with ONNX Runtime
- [ ] A cloud GPU (e.g. L4) and a cloud CPU instance: cost per 1,000 searches/s
      (needs a cloud account)
- [ ] Fail at startup when a GPU is requested but not usable (today the
      service falls back to CPU silently)

### 7. Qdrant Cloud

- [ ] Network round trip from the hosting region, single vs batched queries
- [ ] Cluster size (nodes, shards, replicas) for the overall target
- [ ] Index the image vector (it has no HNSW index today, `m=0`)

### 8. Portable measurement tools

So any machine or cloud gives trustworthy numbers and settings with little
manual work. The current tools work correctly on the laptop; this is about
running them elsewhere and changing their inputs without editing them. Built
in stages, when each piece is first needed; the current tools keep working
meanwhile.

**Design**

- Inputs, not fixed values. One settings file per environment (laptop, cloud
  GPU VM, Kubernetes, Qdrant Cloud) holds everything that differs between
  them: service address and TLS, Qdrant address and API key, collection
  names, results folder, how to restart the service, and where to read
  CPU/GPU from. Every tool reads it; any value can be overridden on the
  command line for one run. Secrets such as the API key come from environment
  variables named in the file, never from the file itself.
- Small modules with one job each, shared by all tools:

| Module | Job | Kinds |
| -- | -- | -- |
| Settings | Load an environment's settings file and apply command-line overrides | – |
| Load runner | Run a k6 test type at given rates and return the run id | k6 in Docker (today's), k6 installed locally, k6 operator |
| Resource readers | CPU, memory and GPU use of the service and of Qdrant over a run | local process (`/proc`), Docker container (cgroup), `nvidia-smi`, Prometheus query, Qdrant `/metrics` |
| Service control | Restart the service with a set of environment variables and wait until healthy | local process, `docker compose`, `kubectl` |
| Result readers | Turn raw output into numbers | per-window latency (`summarize_load_test.py`), profile (`summarize_profile.py`), accuracy and Qdrant cost (`measure_qdrant_search.py`) |
| Test data | Make the test collections available in a Qdrant | build locally (`build_load_test_collection.py`), copy by Qdrant snapshot |

- Tools become thin combinations of modules: the settings sweep is service
  control + load runner + resource readers + result readers, repeated for each
  setting combination.
- A new environment needs a new settings file and, at most, one new module
  kind (for example "restart through `kubectl`"); nothing else changes.

**Hard-coded today, to become inputs**

| Where | Value |
| -- | -- |
| `benchmarks/vector_service/quality/measure_qdrant_search.py` | Qdrant batch size 32, query sample size 150 and seed 54, query and output paths, collection `anime_accuracy_test`, container `echora-dev-qdrant`, CPU read from a local Docker cgroup path |
| `benchmarks/vector_service/diagnostics/profile_service.sh` | 120 s ramp and 15 s settle (tied to the `load` test's shape), default target `localhost:8001` |
| `benchmarks/vector_service/load/run_vector_search.sh` | container `echora-dev-vector-service`, GPU from local `nvidia-smi`, results folder |
| `benchmarks/vector_service/reports/summarize_load_test.py` | results folder |
| `benchmarks/vector_service/diagnostics/run_timed_vector_service.py` | 10 s report interval; wraps the service's functions, so only for a service run as a local process |

`benchmarks/vector_service/load/vector_search.js` already takes all its inputs as variables.

**Work, in stages**

- [ ] Now: design written here
- [ ] With step 6's CPU re-measurement and model-server comparisons (many
      runs with different settings): settings file and loader, resource
      readers moved out of the scripts, service control, then the settings
      sweep; shell wrappers' sampling moves into Python
- [ ] With the cloud work (steps 6–7): Prometheus and Qdrant `/metrics`
      readers, copying test collections by snapshot, queue wait / batch size /
      batch call time as the service's own OpenTelemetry metrics (replaces
      `benchmarks/vector_service/diagnostics/run_timed_vector_service.py`)
- [ ] Pick `RATE` / `MAX_RATE` automatically from a short coarse ramp

### 9. Smaller open items

- [ ] The ~45 ms Qdrant call at 1 search/s over HTTP (see "Found along the
      way")
- [ ] Measure image search (OpenCLIP) the same way as text search
- [ ] Check that indexing does not slow search when both run in one instance
      (they share `EMBED_MAX_CONCURRENCY` and the GPU)
- [ ] `libs/qdrant_db/src/qdrant_db/client.py` is over 500 lines; split it

### 10. Write up

- [ ] Findings complete in ECHO-54
- [ ] Implementation issues for each change worth making

## Configuration

Every setting studied, what the code does by default and what to set. Code
defaults keep main's behaviour; tuned values are set through environment
variables per deployment. `docker/docker-compose.dev.yml` sets the
recommended values; `docker/docker-compose.prd.yml` sets none yet, since
batch sizes depend on the hardware measured in step 6.

| Setting | Code default | Recommended | Why | Evidence |
| -- | -- | -- | -- | -- |
| Qdrant client `cloud_inference` | `True` (in code) | keep | Skips a check that cost ~1.2 ms CPU per query; the service always sends vectors | ECHO-54 finding 22 |
| `EMBED_BATCH_MAX_SIZE` | 1 (off) | 64 | Encodes concurrent queries in one model call; 128 gave no more | findings 15, 24 |
| `EMBED_BATCH_MAX_WAIT_MS` | 0 | 0 | A lone search is not delayed; not tuned yet (step 2) | finding 15 |
| `EMBED_MAX_CONCURRENCY` | 2 | 1 on one GPU | Parallel model calls on one GPU only add waiting: p50 at 500/s 82 ms with 1, 102 with 2. Not measured on CPU instances or for indexing, which uses the same setting | finding 24 |
| `QDRANT_QUERY_BATCH_MAX_SIZE` | 1 (off) | 8 | Sends concurrent searches in one `query_batch_points` call. Qdrant runs one batch's searches one after another per segment, so smaller batches finish sooner | findings 17, 24 |
| `QDRANT_QUERY_BATCH_MAX_WAIT_MS` | 0 | 0 | As above | finding 17 |
| `QDRANT_QUERY_BATCH_CONCURRENCY` | 4 | 16 | With batches of 8, lets Qdrant search several batches on separate threads | finding 24 |
| `QDRANT_PREFER_GRPC` | `false` | `true` when callers ask for IDs only | 30–40% lower latency without payloads; slower with payloads | findings 7, 18 |
| `QDRANT_GRPC_PORT` | 6334 | 6334 | Qdrant and Qdrant Cloud default | finding 18 |
| `QDRANT_SEARCH_RESCORE` | unset | `true` | Hybrid recall 0.917 → 0.958 with oversampling | findings 19–20, step 3F |
| `QDRANT_SEARCH_OVERSAMPLING` | unset | 4.0 | Rescoring needs extra candidates in a merged segment | step 3F |
| `QDRANT_SEARCH_HNSW_EF` | unset | unset | `ef` 512 with 2× oversampling gives 0.001 more hybrid recall for 7% more Qdrant CPU | step 3F |
| `QDRANT_PREFETCH_LIMIT_MULTIPLIER` | 10 | 10 | Fewer candidates lose accuracy for little saving | finding 21 |
| `with_payload` on `SearchRequest` | payloads returned | `false` from the Rust backend | The backend reads records from PostgreSQL | findings 5, 16 |
| `default_segment_number` | 4 | 4 for now | 2 segments cost less CPU but need oversampling; re-measure on real production-sized data | finding 20 |
| `text_vector` HNSW `m` / `ef_construct` | 64 / 256 | keep | Not the accuracy limit | step 3E |

Found along the way, not changed:

- `QDRANT_ENABLE_QUANTIZATION` only controls the collection-wide
  quantization setting; the per-vector settings quantize `text_vector` and
  `image_vector` (int8) whatever it says, so `anime_database` is quantized
  with the flag `false`
- `hnsw_config` in `QdrantConfig` has an `ef` per priority that nothing reads
- qdrant-client's `query_batch_points` deep-copies every request
  (`_resolve_query_batch_request`), with no option to skip it: ~0.06 ms per
  search, too small to work around
- In a smoke test at 1 search/s over HTTP, the Qdrant call took ~45 ms inside
  the service against ~6 ms for the same kind of query sent to Qdrant
  directly; not explained yet (under load over gRPC the whole search takes
  13–15 ms)

## Open decisions

- The p95 latency goal for one search: decides what "handles X searches/s"
  means
- A labelled query set (step 3): which results are correct for real queries
  needs human judgement; needed to measure relevance, not just agreement
  with exact search, and before tuning RRF weights
- A cloud account for step 6
- The hosting region for step 7
