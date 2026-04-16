import json
import os
import sys
from collections import defaultdict


def load_jsonl(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f).get("data", [])


def save_jsonl(anime_list: list[dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in anime_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def smart_merge(base: dict, other: dict) -> dict:
    """Merge `other` into `base`, returning the enriched `base`."""
    # Simple fields — prefer non-empty, status prefers more specific value
    for field in ("picture", "thumbnail", "status", "description"):
        val_a = base.get(field)
        val_b = other.get(field)
        if val_a in (None, "", "UNKNOWN") and val_b not in (None, "", "UNKNOWN"):
            base[field] = val_b
        elif (
            field == "status"
            and val_a == "UNKNOWN"
            and val_b in ("FINISHED", "UPCOMING", "ONGOING")
        ):
            base[field] = val_b

    # Numeric fields — prefer max / non-zero
    for field in ("episodes",):
        val_a = base.get(field) or 0
        val_b = other.get(field) or 0
        base[field] = max(val_a, val_b)

    # List fields — union and deduplicate
    for field in (
        "sources",
        "synonyms",
        "tags",
        "producers",
        "studios",
        "genres",
        "relatedAnime",
    ):
        list_a = base.get(field, [])
        list_b = other.get(field, [])
        base[field] = sorted(set(list_a) | set(list_b))

    # animeSeason — fill in missing year or season from other
    season_a = base.get("animeSeason", {})
    season_b = other.get("animeSeason", {})
    if season_a.get("year") is None and season_b.get("year") is not None:
        season_a["year"] = season_b["year"]
    if season_a.get("season") in (None, "UNDEFINED") and season_b.get("season") not in (
        None,
        "UNDEFINED",
    ):
        season_a["season"] = season_b["season"]
    base["animeSeason"] = season_a

    return base


def merge_database(file_path: str, output_path: str) -> None:
    print(f"Loading {file_path}...")
    anime_list = load_jsonl(file_path)
    initial_count = len(anime_list)
    print(f"Initial entries: {initial_count:,}")

    indices_to_remove: set[int] = set()
    real_seasons = {"WINTER", "SPRING", "SUMMER", "FALL"}

    def _merge_group(indices: list[int]) -> None:
        indices.sort()
        base_idx = indices[0]
        for dup_idx in indices[1:]:
            anime_list[base_idx] = smart_merge(
                anime_list[base_idx], anime_list[dup_idx]
            )
            indices_to_remove.add(dup_idx)

    # ── Tier 1: title + type + year + season (all match, including UNDEFINED) ─
    tier1: dict = defaultdict(list)
    for i, entry in enumerate(anime_list):
        s = entry.get("animeSeason", {})
        year, season = s.get("year"), s.get("season")
        has_real_year = year is not None
        has_real_season = season in real_seasons
        if has_real_year and has_real_season:
            tier1[(entry.get("title"), entry.get("type"), year, season)].append(i)
        elif not has_real_year and not has_real_season:
            tier1[(entry.get("title"), entry.get("type"), year, season)].append(i)

    t1_count = sum(len(v) - 1 for v in tier1.values() if len(v) > 1)
    for indices in tier1.values():
        if len(indices) > 1:
            _merge_group(indices)
    print(f"Tier 1 merged: {t1_count} redundant entries")

    # ── Tier 2: title + type + year (season unknown) ──────────────────────────
    active = [i for i in range(len(anime_list)) if i not in indices_to_remove]
    tier2: dict = defaultdict(list)
    for i in active:
        entry = anime_list[i]
        s = entry.get("animeSeason", {})
        year = s.get("year")
        season = s.get("season")
        if year is not None and season not in real_seasons:
            tier2[(entry.get("title"), entry.get("type"), year)].append(i)

    t2_count = sum(len(v) - 1 for v in tier2.values() if len(v) > 1)
    for indices in tier2.values():
        if len(indices) > 1:
            _merge_group(indices)
    print(f"Tier 2 merged: {t2_count} redundant entries")

    # ── Tier 3: title + type + season (year missing) ──────────────────────────
    active = [i for i in range(len(anime_list)) if i not in indices_to_remove]
    tier3: dict = defaultdict(list)
    for i in active:
        entry = anime_list[i]
        s = entry.get("animeSeason", {})
        year = s.get("year")
        season = s.get("season")
        if year is None and season in real_seasons:
            tier3[(entry.get("title"), entry.get("type"), season)].append(i)

    t3_count = sum(len(v) - 1 for v in tier3.values() if len(v) > 1)
    for indices in tier3.values():
        if len(indices) > 1:
            _merge_group(indices)
    print(f"Tier 3 merged: {t3_count} redundant entries")

    # ── Build final list ──────────────────────────────────────────────────────
    final_list = [e for i, e in enumerate(anime_list) if i not in indices_to_remove]
    total_removed = initial_count - len(final_list)

    print(f"Final entries  : {len(final_list):,}")
    print(f"Total removed  : {total_removed:,}")

    print(f"Saving to {output_path}...")
    save_jsonl(final_list, output_path)
    print("Done.")


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/qdrant_storage/anime-offline-database-minified.json"
    )
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{base}_merged.jsonl"
    merge_database(file_path, output_path)
