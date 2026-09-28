#!/usr/bin/env python3
"""Compare hybrid search results for queries embedded with two model settings.

Embeds every query in ``load/search_queries.json`` with BGE-M3 twice, once per
setting (texts per model pass, token budget per pass), runs the same hybrid
search (dense + sparse prefetch, RRF) for each on the environment's accuracy
collection, and prints how many queries got the identical top results in the
same order and the share of result IDs the two runs have in common.

The top results' exact order already varies a little between runs, because
fp16 vectors change slightly with the other texts in a batch and near-tied
results swap places. ``--with-baseline`` measures that variation for the
first setting (run twice, batches one smaller, one text at a time), so a
change can be judged against it.

Run from the repository root; needs the GPU the service would use:

    ./pants run benchmarks/vector_service/quality/compare_search_results.py -- \\
      --first-chunk-size 256 --second-chunk-size 32 --with-baseline
"""

import argparse
import random
from dataclasses import dataclass

from common.config.qdrant_config import QdrantConfig
from qdrant_client import QdrantClient, models
from vector_db_interface import SparseVectorData
from vector_processing.embedding_models.text.flagembedding_model import (
    FlagEmbeddingModel,
)

from benchmarks.vector_service.toolkit.query_mix import load_queries
from benchmarks.vector_service.toolkit.result_agreement import (
    ResultAgreement,
    compare_top_results,
)
from benchmarks.vector_service.toolkit.settings import load_environment


@dataclass(frozen=True)
class ModelSetting:
    chunk_size: int
    token_budget: int
    batch_size: int

    def label(self) -> str:
        return f"chunks of {self.chunk_size}, budget {self.token_budget}, batches of {self.batch_size}"


@dataclass(frozen=True)
class HybridSearch:
    client: QdrantClient
    collection: str
    dense_vector: str
    sparse_vector: str
    prefetch: int
    rrf_k: int
    limit: int

    def top_ids(self, dense: list[float], sparse: SparseVectorData) -> list[object]:
        prefetch = [
            models.Prefetch(query=dense, using=self.dense_vector, limit=self.prefetch),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse["indices"], values=sparse["values"]
                ),
                using=self.sparse_vector,
                limit=self.prefetch,
            ),
        ]
        hits = self.client.query_points(
            self.collection,
            prefetch=prefetch,
            query=models.RrfQuery(rrf=models.Rrf(k=self.rrf_k)),
            limit=self.limit,
        ).points
        return [hit.id for hit in hits]


def embed(
    model: FlagEmbeddingModel, texts: list[str], setting: ModelSetting
) -> tuple[list[list[float]], list[SparseVectorData]]:
    model._chunk_size = setting.chunk_size
    model._max_tokens_per_pass = setting.token_budget
    dense: list[list[float]] = []
    sparse: list[SparseVectorData] = []
    for start in range(0, len(texts), setting.batch_size):
        batch_dense, batch_sparse = model.encode_with_sparse(
            texts[start : start + setting.batch_size]
        )
        dense += batch_dense
        sparse += [vector for vector in batch_sparse if vector is not None]
    return dense, sparse


def search_all(
    model: FlagEmbeddingModel,
    search: HybridSearch,
    texts: list[str],
    setting: ModelSetting,
) -> list[list[object]]:
    dense, sparse = embed(model, texts, setting)
    return [
        search.top_ids(dense_vector, sparse_vector)
        for dense_vector, sparse_vector in zip(dense, sparse, strict=True)
    ]


def print_agreement(label: str, agreement: ResultAgreement) -> None:
    print(
        f"{label:70} identical {agreement.identical:5}/{agreement.total}  overlap {agreement.overlap:.4f}",
        flush=True,
    )


def build_search(args: argparse.Namespace) -> HybridSearch:
    environment = load_environment(args.environment, args.set)
    qdrant_config = QdrantConfig()
    client = QdrantClient(
        url=environment.qdrant.url, api_key=environment.qdrant.api_key()
    )
    return HybridSearch(
        client=client,
        collection=environment.collections.accuracy,
        dense_vector=qdrant_config.primary_text_vector_name,
        sparse_vector=qdrant_config.primary_sparse_vector_name,
        prefetch=args.prefetch,
        rrf_k=args.rrf_k if args.rrf_k is not None else qdrant_config.rrf_k,
        limit=args.limit,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--environment", default="laptop")
    parser.add_argument(
        "--set", action="append", default=[], help="override, e.g. qdrant.url=..."
    )
    parser.add_argument("--first-chunk-size", type=int, default=256)
    parser.add_argument("--first-token-budget", type=int, default=0)
    parser.add_argument("--second-chunk-size", type=int, default=32)
    parser.add_argument("--second-token-budget", type=int, default=0)
    parser.add_argument(
        "--batch-size", type=int, default=41, help="texts per model call, as under load"
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--prefetch", type=int, default=100, help="candidates per branch"
    )
    parser.add_argument(
        "--rrf-k", type=int, default=None, help="default: QdrantConfig's"
    )
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--with-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    search = build_search(args)
    texts = [text for kind_texts in load_queries().values() for text in kind_texts]
    random.Random(args.seed).shuffle(texts)  # noqa: S311 - repeatable order, not security
    print(f"{len(texts)} queries, collection {search.collection}", flush=True)
    model = FlagEmbeddingModel("BAAI/bge-m3")
    first = ModelSetting(
        args.first_chunk_size, args.first_token_budget, args.batch_size
    )
    second = ModelSetting(
        args.second_chunk_size, args.second_token_budget, args.batch_size
    )
    first_results = search_all(model, search, texts, first)
    if args.with_baseline:
        baselines = [
            ("the same, run again", first),
            (
                f"batches of {args.batch_size - 1}",
                ModelSetting(
                    first.chunk_size, first.token_budget, max(1, args.batch_size - 1)
                ),
            ),
            (
                "one text at a time",
                ModelSetting(first.chunk_size, first.token_budget, 1),
            ),
        ]
        for label, setting in baselines:
            print_agreement(
                f"baseline: {first.label()} vs {label}",
                compare_top_results(
                    first_results, search_all(model, search, texts, setting)
                ),
            )
    second_results = search_all(model, search, texts, second)
    print_agreement(
        f"{first.label()} vs {second.label()}",
        compare_top_results(first_results, second_results),
    )


if __name__ == "__main__":
    main()
