"""Sample search texts in the same mix as the k6 load test.

``load/search_queries.json`` holds three kinds of query (short descriptions,
titles, long descriptions); the load test picks a kind by weight, then a text
of that kind. Tools that run the model or Qdrant directly use the same mix, so
their numbers match what the service sees under load.
"""

import json
import random
from pathlib import Path

from benchmarks.vector_service.toolkit.settings import REPOSITORY_ROOT

QUERY_FILE = (
    REPOSITORY_ROOT / "benchmarks" / "vector_service" / "load" / "search_queries.json"
)
QUERY_KINDS = ("short", "title", "long")
DEFAULT_WEIGHTS = "40,40,20"


class QueryWeightsError(ValueError):
    def __init__(self, text: str) -> None:
        super().__init__(
            f"query weights must be three non-negative numbers for "
            f"{', '.join(QUERY_KINDS)} with a positive sum, got {text!r}"
        )


def parse_query_weights(text: str) -> dict[str, int]:
    """Weights in the load test's ``QUERY_MIX`` form, e.g. ``40,40,20``."""
    parts = text.split(",")
    if len(parts) != len(QUERY_KINDS) or not all(
        part.strip().isdigit() for part in parts
    ):
        raise QueryWeightsError(text)
    weights = dict(zip(QUERY_KINDS, (int(part) for part in parts), strict=True))
    if not sum(weights.values()):
        raise QueryWeightsError(text)
    return weights


def load_queries(path: Path = QUERY_FILE) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def sample_query_mix(
    queries: dict[str, list[str]],
    weights: dict[str, int],
    count: int,
    random_generator: random.Random,
) -> list[str]:
    """``count`` texts, each kind chosen by weight and each text at random."""
    kinds = [kind for kind in weights if weights[kind]]
    chosen_kinds = random_generator.choices(
        kinds, weights=[weights[kind] for kind in kinds], k=count
    )
    return [random_generator.choice(queries[kind]) for kind in chosen_kinds]
