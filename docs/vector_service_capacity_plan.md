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
- Scope: the vector service and Qdrant (hosting still to decide: an EU-based
  provider, see "Open decisions"). The Rust
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
| GPU, plus texts per model pass: chunks of 32 | 1.3M points | ~710 searches/s | per window: p50 74–76 ms at ~450/s; at a steady 600/s p50 / p95 173 / 244 ms | the model thread (~83% busy at 600/s); Python's GIL 20–28% of each model call (findings 29–32) |
| GPU, service and Qdrant on separate cores | 1.3M points | ~790 (no budget) to ~833 (budget 512) searches/s | at a steady 600/s p50 / p95 79 / 114 ms; at ~728/s p50 120–185 ms | likely Qdrant on the laptop's 8 slower cores (finding 33) |

CPU only, every change above, in a container with a CPU limit (laptop's
Ryzen AI 9 HX 370; breakpoint test, stopped when p95 passes 2 s):

| CPU limit | PyTorch threads | Keeps up to | p50 across the run | Memory under load |
| -- | -- | -- | -- | -- |
| 2 CPUs | PyTorch's own (12) | ~3 searches/s | 2,240 ms | 4.3 GiB |
| 2 CPUs | 2 (`OMP_NUM_THREADS=2`) | ~6 searches/s | 580 ms | 2.9–4.4 GiB |
| 4 CPUs | 4 | ~12 searches/s | 265 ms | 4.4 GiB |
| 8 CPUs | 8 | ~20 searches/s | 310 ms | 5.0 GiB |

About 2.5–3 searches/s per CPU; one search alone takes ~200 ms. The
service needs at least 6 GB of memory: with 4 GB it is killed while loading
the models (steady use ~4.1 GiB, of which ~1.8 GiB is model files kept in
the page cache).

Latency in the GPU rows above the last one is k6's Prometheus figure, which
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

Remaining work, in the suggested order: step 6's CPU-only re-measurement,
then step 6's model server and ONNX comparisons, each compared with the
settings sweep (step 8, built). Step 5, the query cache, is on hold. The
smaller items in step 9 fit in between. Step 6's cloud part and step 7 wait
for a cloud account and region.

### 4. Several worker processes per instance

- [ ] On hold: only needed if the Python side becomes the limit. It is not:
      at a steady 500/s the event loop thread is ~42% busy and query
      embedding is the limit (ECHO-54 finding 24)

### 5. Query-embedding cache

On hold (decision 2026-09-27; revisit later). A search's text is embedded
before Qdrant sees anything, and embedding is the limit, so only a cache that
skips the model call helps. What was found:

- An exact-text cache (with light normalisation such as trimming spaces) is
  the only kind that skips the model: web-search logs show 20–40% of queries
  are exact repeats (Markatos, Excite: 20–30%; Xie & O'Hallaron, CMU: 30–40%;
  Teevan et al., Yahoo: 33% repeated by the same user, ~18% across users).
  Anime search may differ; the real rate needs real traffic
  (`echora_embedding_cache_total`).
- A semantic cache (matching paraphrases by embedding similarity, e.g.
  GPTCache) cannot skip the model: looking up a paraphrase needs the new
  query's embedding first. It could only skip Qdrant, and it risks returning
  wrong results for close but different wording. A paraphrase already gets a
  nearly identical embedding and so nearly the same results.
- Today text search never uses the existing embedding cache
  (`encode_text_with_sparse`; finding 3), and the cache stores dense vectors
  only.

If picked up: exact-text cache holding dense and sparse vectors, checked
before the model queue; measured with a sweep on the normal query set
(repeats) and `UNIQUE=true` (no repeats); lowercasing only if the accuracy
tool shows unchanged results. A result cache (text + filters → IDs) could
also skip Qdrant but needs expiry when data changes.

### 6. Hardware and hosting

- [x] Re-measure a CPU-only instance with every change so far (see "Current
      numbers"; sweeps `cpu_threads` and `cpu_scaling`). PyTorch sizes its
      thread pool from the host's cores, not the container's CPU limit
      (PyTorch issues #64864, #193859; vLLM PR #34462 saw ~30% lost
      throughput from the same thing): 12 threads in a 2-CPU container here.
      Setting `OMP_NUM_THREADS` to the CPU limit doubled throughput
- [x] On CPU: `EMBED_MAX_CONCURRENCY` and batch size (sweep `cpu_embedding`,
      4 CPUs): every combination peaked at 8–14 searches/s, within run-to-run
      noise; two model calls with 4 threads each was worst (8/s). Keep batch 64,
      one call at a time
- [x] Production compose: one NVIDIA GPU (`ENABLE_GPU=true`, device
      reservation; the image's torch is the CUDA 13.0 build from `uv.lock`, so
      the host needs driver 580 or newer), memory limit 6G (reservation 5G),
      `OMP_NUM_THREADS=2` for its 2-CPU limit, and every recommended setting
      from the configuration table. `QDRANT_PREFER_GRPC` stays `false` until
      callers ask for IDs only. Values were measured on the laptop GPU;
      re-check them on the production GPU with the sweep
- The dev image installs CPU-only torch (`Dockerfile.dev`), so the benchmark's
  Docker kind with `echora-vector-service:dev` cannot use a GPU; GPU runs use
  the local process kind, or the production image
- [x] Which batching model servers return BGE-M3's sparse output: TEI no
      (dense only; PR #899 open), Infinity and Xinference no, vLLM only as a
      separate request (the model would run twice per search), Triton with an
      ONNX export yes, m3serve yes (an in-process library, not a server)
- [x] Where a batch's time goes on the GPU (batch of 64, query mix): model
      pass 78 ms, tokenizing 2 ms, converting results 2 ms. Overlapping them,
      as m3serve does, can gain ~5% at most
- [ ] Split each batch into length-sorted chunks: 73% of the tokens in a
      batch of 64 are padding. Chunks of 16: 770 → 1,370 texts/s on the model
      alone (32: 1,120; 8: 1,100), vectors unchanged within fp16 rounding
      (dense cosine ≥ 0.9994, sparse tokens same in 1,277 of 1,280). Setting
      `EMBED_MODEL_CHUNK_SIZE` (default 256: one pass per query batch).
      End to end (sweep `model_chunks`, GPU breakpoint): most searches/s
      256 → ~620, 32 → ~710, 16 → ~630; latency at the same rate unchanged
      (p50 74–76 ms at ~450/s in all three)
- [x] Where searches wait at a steady 600/s (`--timed`, sweep
      `model_chunks_steady`; chunks of 32 against 256): p50 174 against
      242 ms. Per search: model wait ~104 against ~154 ms, Qdrant wait ~63
      against ~72 ms. The model thread is still the limit: busy ~83% at 600/s
      (calls of ~57 ms on batches of ~41 texts). The GPU reads only 36–51%
      busy because a pass is many small steps started from Python. Real
      batches are ~41 texts, not 64; with fixed chunks of 32 one chunk still
      holds every long query, so a batch of 41 takes 48 ms alone against
      57 ms for 64. The event loop thread (45% busy) slows each model call by
      ~13% through Python's GIL (48 → 55 ms in a test with a second busy
      thread). The laptop's CPU was 40–50% busy; Qdrant (~54 ms per batch of
      8, ~4 of 16 batches in flight) was not queueing
- [x] Group each batch by length with a token budget per pass
      (`EMBED_MODEL_MAX_TOKENS_PER_PASS`, off by default; sweeps
      `token_budget_breakpoint` / `token_budget_load`). Model alone, batches
      of 41: chunks of 32 884 texts/s, budget 256 1,424, 384 1,211, 512 1,279.
      End to end it does not help: most searches/s no budget ~707, 256 ~613,
      384 ~650, 512 ~683. Inside the service at 600/s a pass with budget 256
      costs ~2× what it costs alone (46 texts in 63 ms against ~30 ms), against
      ~1.2× for chunks of 32: every extra pass pays Python overhead that grows
      under load (Python's GIL, the laptop CPU slowing when busy). Kept, off,
      to try again after the next item
- [x] Fewer Python steps per pass: `torch.compile` and CUDA graphs
      (`mode="reduce-overhead"`, lengths padded to multiples of 16 and chunks
      to 8/16/32 texts). Batch of 41, alone / with a busy Python thread:
      chunks of 32 eager 48.1 / 51.7 ms, compiled 46.6 / 48.2, CUDA graphs
      56.0 / 58.4; budget 256 eager 28.1 / 33.2, compiled 27.4 / 30.4, CUDA
      graphs 46.6 / 50.5 (the fixed sizes add padding back). Vectors equal
      within fp16 rounding (cosine ≥ 0.9994). Not adopted: 2–9% at best
- [x] Why a model call is slower inside the service at 600/s (per-thread
      scheduler counters in `run_timed_vector_service.py`): no budget, 55 ms
      = 43 on a CPU + 11 blocked (mostly Python's GIL) + 0.7 waiting for a
      free core; budget 256, 89 ms = 61 on a CPU (~42 alone) + 25 blocked + 2
      waiting. So the GIL costs 20–28% of each call, and the model thread runs
      ~45% slower while on a CPU, likely because Qdrant's ~6 busy cores share
      the laptop's cores and power budget (in production Qdrant is to run on
      separate machines)
- [x] How much of that is the laptop: service pinned to the fast cores
      (0–3, 12–15, up to 5.2 GHz) with `taskset`, the dev Qdrant container to
      the slower ones (4–11, 16–23, up to 3.3 GHz) with `docker update
      --cpuset-cpus`, restored afterwards. At 600/s: p50 / p95 173 / 244 →
      79 / 114 ms (no budget), 672 / 873 → 88 / 130 ms (budget 256); a model
      call 55 → 14 ms (batches of ~10 instead of ~41), blocked 11 → 2.5 ms.
      Breakpoint: most searches/s no budget ~791, budget 256 ~803, budget
      512 ~833; p50 at ~728/s 185, 121, 120 ms. Unpinned, the OS ran the
      model thread on slow cores next to a busy Qdrant. Now the likely limit
      is Qdrant on the 8 slower cores, a limit of the laptop test. In
      production Qdrant is to run on its own nodes, closer to the pinned case.
      Search results with budget 512 (`quality/compare_search_results.py`):
      top-10 overlap with one pass 0.974, within today's variation
      (0.975–0.991)
- [ ] The model in its own process, to remove the GIL share (now 2.5–3.4 ms
      of a 14–17 ms call when pinned)
- [ ] ONNX Runtime / TensorRT for a faster model pass (a new dependency)
- [x] Search results with chunks of 32 (2,021 real queries, hybrid top 10
      on `anime_accuracy_test`): overlap with one pass 0.977, the same as
      today's own variation from batch makeup (batches of 41 vs one text at a
      time: 0.975; the same batches run twice: 0.990, only 40% identical in
      order, near-tied ranks swap)
- [ ] A model server only if it beats the above: Triton + ONNX (fp16,
      TensorRT) for a faster model pass
- [ ] CPU inference with ONNX Runtime
- [ ] A cloud GPU (e.g. L4) and a cloud CPU instance: cost per 1,000 searches/s
      (needs a cloud account)
- [x] Fail at startup when a GPU is requested but not usable: with
      `ENABLE_GPU=true` the vector service stops with `GpuUnavailableError`,
      naming a CPU-only PyTorch build or a missing CUDA device, and otherwise
      logs the GPU it uses

**Sizing for the overall target** (estimates from measured per-instance
numbers; the cloud machines themselves are not measured yet)

| Option | Per instance | Instances for 10,000 / 20,000 searches/s | On-demand price (AWS us-east-1) |
| -- | -- | -- | -- |
| CPU only | ~2.5–3 searches/s per CPU | ~3,500–4,000 / ~7,000–8,000 CPUs | not viable |
| GPU, RTX 4070 Laptop as measured | ~590 searches/s | ~17 / ~34 | – |
| NVIDIA T4, `g4dn.xlarge` (4 vCPU, 16 GiB, 16 GB GPU) | not measured | ~17 / ~34 if like the laptop | $0.526/h (~$384/month each) |
| NVIDIA L4, `g6.xlarge` (4 vCPU, 16 GiB, 24 GB GPU) | not measured | ~17 / ~34 if like the laptop | $0.805/h (~$588/month each) |

- Published numbers, other setups: BGE-M3 on a T4 with a batching server
  (m3serve) reached ~1,600–1,900 texts/s at batches of 64–128; the same
  model on 1–2 cloud vCPUs, ~0.6–0.8 requests/s (nullmirror, llama.cpp);
  guides put CPU at "fine below ~10 searches/s" and recommend a GPU above
  that (Jina sizing guide), with the L4 as the usual choice for BGE-M3-class
  models (Jina, Superlinked).
- Memory per instance: at least 6 GB for the service (both models load at
  start); a GPU with 4 GB or more holds BGE-M3 (~2 GB in FP16).
- Qdrant memory, from Qdrant's capacity-planning formula, per million
  points: `text_vector` 1024 × 4 bytes ≈ 4.1 GB, its int8 copy ≈ 1 GB,
  HNSW (`m` 64) ≈ 0.6 GB, plus sparse vectors, image vectors and payloads.

### 7. Qdrant in production

- [ ] Run Qdrant on its own nodes, apart from the GPU vector service nodes
      (finding 33: sharing a machine's CPU slows the service)
- [ ] Network round trip between the service and Qdrant nodes at the chosen
      provider, single vs batched queries
- [ ] Cluster size (nodes, shards, replicas) for the overall target
- [x] Image vector without an index (`m=0`): correct, since HNSW cannot index
      MaxSim multivectors (Qdrant docs); every image search scores every image
      vector that passes its filter (ECHO-54 finding 34)
- [x] Two-stage image search, accuracy on real images (finding 38,
      `quality/measure_image_two_stage.py`): each point also gets one indexed
      main vector; stage 1 finds candidates by it, stage 2 compares the query
      with every image of those candidates (Qdrant's documented pattern: an
      indexed mean-pooled vector for candidates, the original multivector to
      rerank; for ColPali Qdrant reports 13x faster retrieval with near
      identical quality using mean pooling). Test data: 1,921 anime,
      characters and episodes with 4,930 real images from the local
      enrichment data (all providers, AniDB included), one image of each
      entity with two or more held out as the query (935 queries). Right
      entity in top 10: today 69.6%; average main 69.3% (20 candidates),
      69.7% (50), 69.8% (100); first image as main 62.5% (100). The average
      keeps today's answers: for the 651 queries today answers, stage 1 ranks
      the right entity within 9 for 95% and within 28 for 99%. The first
      image does not (90% within 200). Choice: the average of a point's
      images, not a chosen main image
- [x] Same check at 41,013 points (finding 41): the offline database's 39,092
      anime covers added as one-image wrong answers. Stage 1 with the average
      vector, for queries today answers: 95% within 9, 99% within 28, the
      same as at 1,921 points. Right entity 1st / top 10: today 53.2 / 69.0%,
      average main with 20 candidates 53.3 / 68.9%, with 100 53.3 / 69.0%;
      first image with 100 47.1 / 61.8%. With CCIP on the top 50, characters
      52.9 / 68.5% → 58.6 / 74.0% (average main). Candidate count: 100 in
      stage 1 leaves a wide margin over rank 28. Caveat: the added wrong
      answers are anime covers; production adds ~770k characters, whose
      pictures look more alike, so the stage-1 rank should be re-checked when
      real character images exist at scale
- [x] Two-stage image search cost (finding 43): `anime_image_load_test`
      rebuilt with each point's average image vector, indexed
      (`build_image_load_test_collection.py --average-vector`), measured with
      `measure_image_search.py` (two-stage variant). Qdrant CPU per search /
      most searches/s, today → two-stage with 100 candidates: no filter
      252 ms / 72 → 29 ms / 281; characters 237 ms / 77 → 26 ms / 285; anime
      9.4 ms / 533 → 5.0 ms / 381. With 50 or 20 candidates the cost is the
      same within noise (25–34 ms), so stage 1 (the index search) is most of
      it; with random vectors that is likely pessimistic. Keep 100 candidates.
      One at a time (fastest / average / p99 / slowest): no filter today
      20.1 / 23.7 / 28.6 / 29.6 ms, two-stage 4.9 / 6.3 / 8.1 / 8.4 ms;
      characters today 19.0 / 22.6 / 28.0 / 28.5 ms, two-stage 6.3 / 8.2 /
      9.9 / 10.5 ms
- [ ] If image searches must cover characters or everything at scale:
      implement the average main vector (schema change and re-indexing),
      with the candidate count from the check above
- [ ] Image search accuracy itself (today 53.8% right entity first, 69.6%
      in top 10 on this test; anime 88.5%, characters 52.8%). Found so far,
      not measured here: general CLIP models fit anime illustrations poorly
      (anime-illust-image-searcher); CLIP fine-tuned on anime data does
      better (Anime-2026 dataset paper: P@10 8.56 → 10.21; DanbooruCLIP,
      ViT-L/14 fine-tuned on Danbooru); anime character models such as CCIP
      (contrastive anime character image pre-training); combining image and
      tags (TCSR-Net: R@1 0.94 with full tags, ~1k-item gallery). Research
      (finding 39), to be measured with `quality/measure_image_two_stage.py`:
      - Another image model: SigLIP 2 (one report on fine-grained
        classification: SigLIP2 ~92%, CLIP ViT-L ~59%, DINOv2 ~41%);
        DanbooruCLIP (the same ViT-L/14 as ours, fine-tuned on Danbooru 2021
        and pixiv); fine-tuned DINOv2 did best in an anime-to-manga matcher
        (~81% R@1)
      - CCIP (deepghs): trained to tell whether two single-character anime
        images show the same character (~240k images, 3,982 characters;
        best model F1 0.94, the default one 0.92). It compares images with
        its own learned metric, not cosine, so it fits as a reranker of
        candidates, not as the Qdrant vector; it knows no character names
      - An anime tagger (WD EVA02-Large v3, Danbooru): ratings, character
        and general tags (P=R threshold 0.53, macro F1 0.48); only tags with
        600+ Danbooru images, so minor characters are missing. Tags could
        filter or boost results, or be a sparse vector next to the image
        vector
      - Fine-tuning on our own data: pictures of the same character from
        different providers are ready-made positive pairs for contrastive
        training
- [x] CCIP as a reranker (finding 40, `quality/rerank_with_ccip.py` in its own
      Python 3.12 environment, since `dghs-imgutils` needs numpy 1.x; top 50
      of each search reranked by each candidate's closest image). Characters
      (909 queries), right entity 1st / in top 10: today 52.8 / 69.1% →
      58.1 / 73.6%; average two-stage 52.8 / 69.3% → 58.4 / 74.1%; first
      image 46.6 / 61.7% → 51.6 / 66.7%. Anime (26): 1st 88.5 → 80.8%, top
      10 unchanged (CCIP is built for single-character images, covers are
      not). So CCIP helps character searches only; 17 ms per image on the GPU
- [x] Other image models (finding 42, `--image-model` in
      `quality/measure_image_two_stage.py`; the service's own OpenCLIP class
      stays the default): right entity 1st / in top 10 on the real-image
      test, and stage-1 rank (average vector) holding 99% of today's answers.
      Service OpenCLIP ViT-L/14: 53.8 / 69.6%, 28. SigLIP 2 SO400M (378px):
      51.1 / 70.2%, 157. SigLIP 2 Large (384px): 44.1 / 64.3%, 77.
      DanbooruCLIP: 39.7 / 51.1%, 105 (and no licence stated). The current
      model stays; CCIP reranking is the gain so far

### 8. Portable measurement tools

So any machine or cloud gives trustworthy numbers and settings with little
manual work. The current tools work correctly on the laptop; this is about
running them elsewhere and changing their inputs without editing them. Built
in stages, when each piece is first needed; the current tools keep working
meanwhile.

**Design**

- Inputs, not fixed values. One settings file per environment (laptop, cloud
  GPU VM, Kubernetes, a hosted Qdrant) holds everything that differs between
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

**Built** (`benchmarks/vector_service/`, see its README)

- `environments/laptop.toml` and `toolkit/settings.py`: one file per
  environment, `--set section.field=value` overrides, protected collections
  refused, the Qdrant key only from the variable the file names
- `toolkit/k6_runner.py` + `load/run_load_test.py`: the k6 run, with k6's
  HTML report and millisecond sample timestamps added; `run_vector_search.sh`
  is a wrapper with its old arguments and variables
- `toolkit/resources.py`: local process (`/proc`, including the event loop
  thread), Docker container (`docker stats`, cgroup), `nvidia-smi`; the
  resource CSV keeps its first six columns and adds Qdrant CPU and event loop
  thread CPU
- `toolkit/service_control.py`: a local process or its own container
  (`echora-bench-vector-service`, with CPU/memory/GPU limits), started with a
  run's settings, ready when the gRPC health check reports SERVING; fails a
  run that asks for the GPU but is not on it
- `toolkit/results.py`: per-window latency, steady-part summary, resource
  means, profile grouping; the two summary scripts print exactly what they
  printed before
- `sweep.py` + `sweeps/*.toml`: one comparison table per sweep; it
  reproduced finding 24 (one against two model calls at 500/s: p50 82 against
  106 ms)
- `quality/measure_qdrant_search.py`: `--environment`, `--batch-size`,
  `--sample-size`, `--seed`
- `diagnostics/time_model_passes.py`: the model on its own per batch size,
  chunk size, token budget and runner (eager, `torch.compile`, CUDA graphs),
  optionally with a busy Python thread and a phase split
- `quality/compare_search_results.py`: whether a model setting changes
  hybrid search results, against today's own variation
- `diagnostics/run_timed_vector_service.py` splits each model call into
  running, waiting for a core and blocked (Linux scheduler counters)
- `test_data/build_image_load_test_collection.py` +
  `quality/measure_image_search.py`: image search cost at production size
- `diagnostics/simulate_indexing_load.py` + `sweeps/search_steady.toml`:
  search latency while indexing shares the GPU
- `sweeps/low_load_transport.toml`: HTTP against gRPC at 1 search/s, timed
- `test_data/download_images.py`, `toolkit/image_entities.py`,
  `quality/measure_image_two_stage.py`: real images grouped by entity, and
  today's image search against the two-stage search
- Tests in `benchmarks/vector_service/tests/`

Still fixed: the `load` test's 2 min ramp is mirrored in `sweep.py` and
`diagnostics/profile_service.sh` (overridable there with `RAMP_SECONDS`).

**Work, in stages**

- [x] Design
- [x] Settings file and loader, resource readers, service control (local
      process and Docker), settings sweep; the shell wrapper's sampling moved
      into Python
- [ ] With the cloud work (steps 6–7): `kubectl` service control, Prometheus
      and Qdrant `/metrics` readers, copying test collections by snapshot,
      queue wait / batch size / batch call time as the service's own
      OpenTelemetry metrics (replaces
      `benchmarks/vector_service/diagnostics/run_timed_vector_service.py`)
- [ ] Pick `RATE` / `MAX_RATE` automatically from a short coarse ramp
- [ ] An `extends` option so similar environment files share a base

### 9. Smaller open items

- [x] The ~45 ms Qdrant call at 1 search/s over HTTP no longer happens
      (finding 37, sweep `low_load_transport` with `--timed`): at 1 search/s
      the Qdrant call takes 8.4–8.8 ms over HTTP and 5.9–7.2 ms over gRPC,
      the whole search 16 and 13–14 ms; the code path has changed since
      (inference check skipped, query batching, gRPC). The only outlier was
      the first search after start-up (~230–250 ms): `MODEL_WARM_UP`, which
      production forces on, was read nowhere. The service now runs each model
      once at start-up when it is set, before it reports healthy; the first
      search then takes 17–21 ms and p99 at 1/s fell from 226–252 to
      23–25 ms
- [x] Measure image search (finding 34, `quality/measure_image_search.py`
      on `anime_image_load_test`: 40,346 anime with 1–6 random image vectors
      and 769,693 characters with 1–2, production's image vector settings).
      Qdrant, laptop: no filter p50 31 ms, ~74 searches/s, ~246 ms Qdrant CPU
      per search; anime only 5.5 ms, ~485/s, ~12 ms; characters only 31 ms,
      ~78/s, ~224 ms (hybrid text search: ~9–12 ms, ~1,200/s). Encoding the
      uploaded image with OpenCLIP ViT-L/14: 41–43 ms on the GPU in fp32,
      ~650 ms on 4 CPU threads
- [x] OpenCLIP in fp16 on the GPU (`torch.autocast`, as OpenCLIP's README
      runs inference; finding 35): 256 real images, cosine against fp32 ≥
      0.99986, 99.7% of top-10 neighbours shared; images/s at batch 1 / 8 /
      32: 24 / 29 / 30 → 44 / 85 / 81. Also speeds up indexing
- [ ] Image searches in the service go through one model call at a time
      (`EMBED_MAX_CONCURRENCY`) and are not batched: ~44 images/s per
      instance on this GPU even in fp16
- [x] Search while indexing shares the GPU (finding 36). The vector service
      does not index today (only `scripts/`; the planned consumer that embeds
      on NATS events does not exist yet), so this measures a GPU shared with an
      indexing process (`diagnostics/simulate_indexing_load.py`: 200-token
      texts in batches of 32 plus 8 images per loop, flat out) against the
      `search_steady` sweep at 400/s: search held 400/s, but p50 / p95 went
      from 61 / 86 to 277 / 381 ms; indexing ran ~72 texts/s and ~18 images/s
      meanwhile. Bulk indexing belongs on its own GPU, or throttled / off
      peak; a trickle of updates is a much smaller load
- [ ] `libs/qdrant_db/src/qdrant_db/client.py` is over 500 lines; split it

### 10. Write up

- [ ] Findings complete in ECHO-54
- [ ] Implementation issues for each change worth making

## Configuration

Every setting studied, what the code does by default and what to set. Code
defaults keep main's behaviour; tuned values are set through environment
variables per deployment. `docker/docker-compose.dev.yml` and
`docker/docker-compose.prd.yml` set the recommended values (production on one
NVIDIA GPU); they were measured on the laptop GPU and are re-checked on the
production GPU once deployed.

| Setting | Code default | Recommended | Why | Evidence |
| -- | -- | -- | -- | -- |
| Qdrant client `cloud_inference` | `True` (in code) | keep | Skips a check that cost ~1.2 ms CPU per query; the service always sends vectors | ECHO-54 finding 22 |
| `EMBED_BATCH_MAX_SIZE` | 1 (off) | 64 | Encodes concurrent queries in one model call; 128 gave no more | findings 15, 24 |
| `EMBED_MODEL_CHUNK_SIZE` | 256 (one pass per query batch) | 32 | Model alone 1.8× faster with 16 on batches of 64; end to end ~620 → ~710 searches/s with 32, p50 at 600/s 242 → 174 ms; results within today's variation | findings 29–31 |
| `EMBED_MODEL_MAX_TOKENS_PER_PASS` | 0 (off) | 0 for now; 512 looks best | Unpinned on the laptop: 256 slower (~613 against ~707/s), 512 even. Service and Qdrant on separate cores (closer to production): 512 ~833 against ~791/s, p50 at ~728/s 120 against 185 ms. Single runs; re-check on the production GPU | findings 31, 33 |
| `MODEL_WARM_UP` | false (production forces true) | true | Runs each model once at start-up, so the first searches after a start do not take ~230–250 ms | finding 37 |
| `EMBED_BATCH_MAX_WAIT_MS` | 0 | 0 | A lone search is not delayed; not tuned yet (step 2) | finding 15 |
| `EMBED_MAX_CONCURRENCY` | 2 | 1 on one GPU | Parallel model calls on one GPU only add waiting: p50 at 500/s 82 ms with 1, 102 with 2. Not measured on CPU instances or for indexing, which uses the same setting | finding 24 |
| `QDRANT_QUERY_BATCH_MAX_SIZE` | 1 (off) | 8 | Sends concurrent searches in one `query_batch_points` call. Qdrant runs one batch's searches one after another per segment, so smaller batches finish sooner | findings 17, 24 |
| `QDRANT_QUERY_BATCH_MAX_WAIT_MS` | 0 | 0 | As above | finding 17 |
| `QDRANT_QUERY_BATCH_CONCURRENCY` | 4 | 16 | With batches of 8, lets Qdrant search several batches on separate threads | finding 24 |
| `QDRANT_PREFER_GRPC` | `false` | `true` when callers ask for IDs only | At 300/s: IDs only p50 66 → 46 ms, p95 92 → 65 ms; with payloads 60–65% slower (p50 124 → 201 ms). Applies to every search of the service | findings 7, 18, 25 |
| `QDRANT_GRPC_PORT` | 6334 | 6334 | Qdrant's default | finding 18 |
| `OMP_NUM_THREADS` (CPU instances) | unset: PyTorch uses the host's core count | the container's CPU limit (production compose: 2) | 2 CPUs: ~3 → ~6 searches/s, p50 2,240 → 580 ms | finding 27 |
| Container memory limit | – | at least 6G (production compose: 6G) | 4G is killed while loading the models | finding 27 |
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

## Open decisions

- The p95 latency goal for one search: decides what "handles X searches/s"
  means
- A labelled query set (step 3): which results are correct for real queries
  needs human judgement; needed to measure relevance, not just agreement
  with exact search, and before tuning RRF weights
- Where production runs: an EU-based infrastructure provider (managed Qdrant
  Cloud only runs on AWS, Google Cloud and Azure). Qdrant there either as
  Qdrant Hybrid Cloud (Qdrant runs in your own Kubernetes cluster and is
  managed from the Qdrant Cloud console; needs standard Kubernetes with CSI
  block storage and snapshots, and an outgoing connection to Qdrant Cloud for
  telemetry and management only; OVHcloud Managed Kubernetes is among
  Qdrant's documented platforms; price from Qdrant's sales team), or
  self-hosted open-source Qdrant (free, Apache 2.0; Docker or Helm; upgrades,
  backups and scaling are ours). Hybrid Cloud saves operations work once
  there is Kubernetes and several Qdrant nodes; self-hosted is simpler on a
  few VMs. `docker/docker-compose.prd.yml` is self-hosted today, with Qdrant
  on the same host as the service
- Which EU provider, and its NVIDIA GPU machines for the vector service
  (steps 6 and 7 measure there)
