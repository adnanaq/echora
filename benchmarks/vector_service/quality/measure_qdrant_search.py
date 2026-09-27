#!/usr/bin/env python3
"""Measure how accurate and how costly Qdrant search settings are.

Three steps, run separately:

``queries``
    Embed a fixed sample of the load test queries once with the service's text
    model (150 short, 150 title and every long query, seed 54) and write the
    dense and sparse vectors to ``data/search_quality/queries.json``. The other
    steps reuse them, so they run without the model.

``accuracy``
    For each setting, compare Qdrant's results with an exact search and score
    them with ranx (recall@10 and NDCG@10, overall and per query kind). Two
    measurements: the dense vector alone against exact kNN on full-precision
    vectors, and the service's hybrid query (dense and sparse candidates, RRF
    k=2, top 10) against the same query with an exact dense branch. The exact
    results are cached per collection, measurement and candidate count.

``cost``
    For each setting, run the hybrid query in batches of 32 with 4 batches in
    flight (``--in-flight`` changes it), as the service sends them, and report queries per second and
    Qdrant's own time per query. For a local Qdrant container the CPU it used
    per query is read from its cgroup counter; otherwise that column is empty.

Qdrant's address and API key come from the service settings (``QDRANT_URL``,
``QDRANT_API_KEY``). Run ``accuracy`` against a collection of real points,
since noisy copies lower recall on their own.
"""

import argparse
import json
import random
import subprocess
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from common.config.qdrant_config import QdrantConfig

QUERY_SOURCE = Path("benchmarks/vector_service/load/search_queries.json")
OUTPUT_DIR = Path("data/search_quality")
QUERY_FILE = OUTPUT_DIR / "queries.json"
QUERY_SEED = 54
SAMPLE_SIZE = 150
TOP_K = 10
RRF_K = 2
BATCH_SIZE = 32
DEFAULT_BATCHES_IN_FLIGHT = 4
QUERY_KINDS = ("short", "title", "long")
EXACT_SEARCH = {"exact": True, "quantization": {"ignore": True}}

SEARCH_SETTINGS: dict[str, dict[str, Any]] = {
    "today": {},
    "ef128": {"hnsw_ef": 128},
    "ef512": {"hnsw_ef": 512},
    "rescore": {"quantization": {"rescore": True}},
    "rescore_os2": {"quantization": {"rescore": True, "oversampling": 2.0}},
    "rescore_os4": {"quantization": {"rescore": True, "oversampling": 4.0}},
    "ef128_rescore": {"hnsw_ef": 128, "quantization": {"rescore": True}},
    "ef128_rescore_os2": {
        "hnsw_ef": 128,
        "quantization": {"rescore": True, "oversampling": 2.0},
    },
    "ef512_rescore": {"hnsw_ef": 512, "quantization": {"rescore": True}},
    "ef512_rescore_os2": {
        "hnsw_ef": 512,
        "quantization": {"rescore": True, "oversampling": 2.0},
    },
    "ef1024_rescore": {"hnsw_ef": 1024, "quantization": {"rescore": True}},
    "no_quantization": {"quantization": {"ignore": True}},
}


class UnsupportedUrlSchemeError(ValueError):
    """Raised when QDRANT_URL is not an http or https address."""

    def __init__(self, url: str) -> None:
        super().__init__(f"QDRANT_URL must be http or https, got {url}")


class UnexpectedRanxResultError(TypeError):
    """Raised when ranx returns a single number where a score per metric was asked."""

    def __init__(self, result: object) -> None:
        super().__init__(f"ranx returned {result!r} instead of a score per metric")


class QdrantRestClient:
    def __init__(self, url: str, api_key: str | None, collection: str) -> None:
        if not url.startswith(("http://", "https://")):
            raise UnsupportedUrlSchemeError(url)
        self._url = url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["api-key"] = api_key
        self._collection = collection

    def query_batch(self, bodies: list[dict[str, Any]]) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 - scheme checked in __init__
            f"{self._url}/collections/{self._collection}/points/query/batch",
            data=json.dumps({"searches": bodies}).encode(),
            headers=self._headers,
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 - scheme checked in __init__
            return json.load(response)

    def top_hits(
        self, bodies: list[dict[str, Any]]
    ) -> tuple[list[dict[str, float]], float]:
        hits: list[dict[str, float]] = []
        qdrant_seconds = 0.0
        for start in range(0, len(bodies), BATCH_SIZE):
            response = self.query_batch(bodies[start : start + BATCH_SIZE])
            qdrant_seconds += response["time"]
            hits.extend(
                {str(point["id"]): point["score"] for point in result["points"]}
                for result in response["result"]
            )
        return hits, qdrant_seconds


def dense_query(query: dict[str, Any], search_params: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": query["dense"],
        "using": "text_vector",
        "limit": TOP_K,
        "with_payload": False,
    }
    if search_params:
        body["params"] = search_params
    return body


def hybrid_query(
    query: dict[str, Any], search_params: dict[str, Any], candidates: int
) -> dict[str, Any]:
    dense_branch: dict[str, Any] = {
        "query": query["dense"],
        "using": "text_vector",
        "limit": candidates,
    }
    if search_params:
        dense_branch["params"] = search_params
    sparse_branch = {
        "query": query["sparse"],
        "using": "text_sparse_vector",
        "limit": candidates,
    }
    return {
        "prefetch": [dense_branch, sparse_branch],
        "query": {"rrf": {"k": RRF_K}},
        "limit": TOP_K,
        "with_payload": False,
    }


def build_queries(
    measurement: str,
    queries: list[dict[str, Any]],
    search_params: dict[str, Any],
    candidates: int,
) -> list[dict[str, Any]]:
    if measurement == "dense":
        return [dense_query(query, search_params) for query in queries]
    return [hybrid_query(query, search_params, candidates) for query in queries]


def embed_queries() -> None:
    from vector_processing.embedding_models.text.flagembedding_model import (
        FlagEmbeddingModel,
    )

    source = json.loads(QUERY_SOURCE.read_text())
    random_generator = random.Random(QUERY_SEED)  # noqa: S311 - repeatable sample, not security
    texts = [
        ("short", text)
        for text in random_generator.sample(source["short"], SAMPLE_SIZE)
    ]
    texts += [
        ("title", text)
        for text in random_generator.sample(source["title"], SAMPLE_SIZE)
    ]
    texts += [("long", text) for text in source["long"]]
    model = FlagEmbeddingModel("BAAI/bge-m3")
    dense, sparse = model.encode_with_sparse([text for _, text in texts])
    records = [
        {"kind": kind, "text": text, "dense": vector, "sparse": sparse_vector}
        for (kind, text), vector, sparse_vector in zip(
            texts, dense, sparse, strict=True
        )
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_FILE.write_text(json.dumps(records))
    print(f"embedded {len(records)} queries into {QUERY_FILE}")


def load_queries() -> list[dict[str, Any]]:
    if not QUERY_FILE.exists():
        print(f"{QUERY_FILE} is missing; run the queries step first")
        raise SystemExit(1)
    return json.loads(QUERY_FILE.read_text())


def exact_results(
    client: QdrantRestClient,
    collection: str,
    measurement: str,
    queries: list[dict[str, Any]],
    candidates: int,
) -> list[dict[str, float]]:
    cache = OUTPUT_DIR / f"exact_{collection}_{measurement}_{candidates}.json"
    if not cache.exists():
        bodies = build_queries(measurement, queries, EXACT_SEARCH, candidates)
        hits, _ = client.top_hits(bodies)
        cache.write_text(json.dumps(hits))
    return json.loads(cache.read_text())


def ranx_scores(
    exact: list[dict[str, float]],
    found: list[dict[str, float]],
    query_indexes: list[int],
) -> dict[str, float]:
    from ranx import Qrels, Run, evaluate

    labelled = [index for index in query_indexes if exact[index]]
    qrels = Qrels({f"q{index}": dict.fromkeys(exact[index], 1) for index in labelled})
    run = Run({f"q{index}": found[index] or {"none": 0.0} for index in labelled})
    scores = evaluate(qrels, run, ["recall@10", "ndcg@10"])
    if not isinstance(scores, dict):
        raise UnexpectedRanxResultError(scores)
    return scores


def measure_accuracy(
    client: QdrantRestClient, collection: str, settings: list[str], candidates: int
) -> None:
    warnings.filterwarnings("ignore", module="ranx")
    queries = load_queries()
    indexes_by_kind = {
        kind: [index for index, query in enumerate(queries) if query["kind"] == kind]
        for kind in QUERY_KINDS
    }
    all_indexes = list(range(len(queries)))
    print(f"{collection}: {len(queries)} queries, {candidates} candidates per branch")
    print(
        f"{'measure':7} {'setting':18} {'recall@10':>9} {'ndcg@10':>8}"
        " | recall short / title / long"
    )
    for measurement in ("dense", "hybrid"):
        exact = exact_results(client, collection, measurement, queries, candidates)
        for name in settings:
            bodies = build_queries(
                measurement, queries, SEARCH_SETTINGS[name], candidates
            )
            found, _ = client.top_hits(bodies)
            overall = ranx_scores(exact, found, all_indexes)
            per_kind = " / ".join(
                f"{ranx_scores(exact, found, indexes)['recall@10']:.3f}"
                for indexes in indexes_by_kind.values()
            )
            print(
                f"{measurement:7} {name:18} {overall['recall@10']:9.4f}"
                f" {overall['ndcg@10']:8.4f} | {per_kind}",
                flush=True,
            )


def container_cpu_seconds(container: str) -> float | None:
    inspect = subprocess.run(  # noqa: S603 - fixed docker command, no shell
        ["docker", "inspect", "-f", "{{.Id}}", container],  # noqa: S607 - docker from PATH
        capture_output=True,
        text=True,
        check=False,
    )
    container_id = inspect.stdout.strip()
    stat_file = Path(
        f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope/cpu.stat"
    )
    if inspect.returncode != 0 or not stat_file.exists():
        return None
    for line in stat_file.read_text().splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1]) / 1_000_000
    return None


def run_in_flight(
    client: QdrantRestClient,
    bodies: list[dict[str, Any]],
    rounds: int,
    batches_in_flight: int,
) -> float:
    batches = [
        bodies[start : start + BATCH_SIZE]
        for start in range(0, len(bodies), BATCH_SIZE)
    ] * rounds
    started = time.perf_counter()
    with ThreadPoolExecutor(batches_in_flight) as pool:
        list(pool.map(client.query_batch, batches))
    return time.perf_counter() - started


def measure_cost(
    client: QdrantRestClient,
    settings: list[str],
    candidates: int,
    container: str,
    rounds: int,
    batches_in_flight: int,
) -> None:
    queries = load_queries()
    if container_cpu_seconds(container) is None:
        print(f"no local container {container}: CPU per query not measured")
    print(
        f"{'setting':18} {'Qdrant ms/query':>15} {'queries/s':>9} {'CPU ms/query':>12}"
    )
    for name in settings:
        bodies = build_queries("hybrid", queries, SEARCH_SETTINGS[name], candidates)
        client.top_hits(bodies)
        _, qdrant_seconds = client.top_hits(bodies)
        cpu_before = container_cpu_seconds(container)
        elapsed = run_in_flight(client, bodies, rounds, batches_in_flight)
        cpu_after = container_cpu_seconds(container)
        query_count = len(bodies) * rounds
        cpu_column = ""
        if cpu_before is not None and cpu_after is not None:
            cpu_column = f"{(cpu_after - cpu_before) / query_count * 1000:12.2f}"
        print(
            f"{name:18} {qdrant_seconds / len(bodies) * 1000:15.2f}"
            f" {query_count / elapsed:9.0f} {cpu_column}",
            flush=True,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("queries", help="embed the fixed query sample")
    for step in ("accuracy", "cost"):
        step_parser = steps.add_parser(step)
        step_parser.add_argument("settings", nargs="+", choices=sorted(SEARCH_SETTINGS))
        step_parser.add_argument("--collection", default="anime_accuracy_test")
        step_parser.add_argument("--candidates", type=int, default=100)
    cost_parser = steps.choices["cost"]
    cost_parser.add_argument("--container", default="echora-dev-qdrant")
    cost_parser.add_argument("--rounds", type=int, default=20)
    cost_parser.add_argument("--in-flight", type=int, default=DEFAULT_BATCHES_IN_FLIGHT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.step == "queries":
        embed_queries()
        return
    qdrant_settings = QdrantConfig()
    client = QdrantRestClient(
        qdrant_settings.qdrant_url, qdrant_settings.qdrant_api_key, arguments.collection
    )
    if arguments.step == "accuracy":
        measure_accuracy(
            client, arguments.collection, arguments.settings, arguments.candidates
        )
    else:
        measure_cost(
            client,
            arguments.settings,
            arguments.candidates,
            arguments.container,
            arguments.rounds,
            arguments.in_flight,
        )


if __name__ == "__main__":
    main()
