"""Rerank search candidates by a per-image score, and count how often the right one wins.

Used to test a second model (such as CCIP) on the candidates an image search
returned: each candidate entity gets the score of its closest image, the way
the collection's MaxSim scores an entity by its best-matching image.
"""

from collections.abc import Mapping, Sequence


def rerank_by_best_image(
    candidates: Sequence[str],
    entity_images: Mapping[str, Sequence[str]],
    differences: Mapping[str, float],
) -> list[str]:
    """Candidates ordered by their closest image (smallest difference first).

    Candidates with no scored image keep their original order after the rest.
    """
    best: dict[str, float] = {}
    for key in candidates:
        scores = [
            differences[url] for url in entity_images.get(key, ()) if url in differences
        ]
        if scores:
            best[key] = min(scores)
    scored = sorted(best, key=best.__getitem__)
    return scored + [key for key in candidates if key not in best]


def hit_rates(
    results: Sequence[Sequence[str]], targets: Sequence[str], top: int
) -> tuple[float, float]:
    """Share of queries whose target is first, and within the first ``top``."""
    first = sum(
        list(ranked[:1]) == [target]
        for ranked, target in zip(results, targets, strict=True)
    )
    within = sum(
        target in ranked[:top] for ranked, target in zip(results, targets, strict=True)
    )
    return first / len(targets), within / len(targets)
