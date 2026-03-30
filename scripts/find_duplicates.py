import json
import os
from collections import defaultdict
from urllib.parse import urlparse


def get_provider(url):
    domain = urlparse(url).netloc
    if "myanimelist.net" in domain:
        return "MAL"
    if "anilist.co" in domain:
        return "AniList"
    if "kitsu.app" in domain:
        return "Kitsu"
    if "anidb.net" in domain:
        return "AniDB"
    if "anime-planet.com" in domain:
        return "Anime-Planet"
    if "anisearch.com" in domain:
        return "AniSearch"
    if "livechart.me" in domain:
        return "LiveChart"
    if "ann.com" in domain:
        return "ANN"
    if "animecountdown.com" in domain:
        return "Countdown"
    if "simkl.com" in domain:
        return "Simkl"
    return domain


def find_duplicates(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            anime_list = [json.loads(line) for line in f if line.strip()]
        else:
            anime_list = json.load(f).get("data", [])
    total_count = len(anime_list)
    print(f"Total anime entries: {total_count}")

    # 1. Internal entry duplicates (Multiple links of same provider in one entry)
    internal_duplicates = []
    for idx, entry in enumerate(anime_list):
        sources = entry.get("sources", [])
        providers = [get_provider(s) for s in sources]
        if len(providers) != len(set(providers)):
            internal_duplicates.append(entry)

    # 2. Semantic duplicates — three tiers by confidence
    tier1: dict = defaultdict(list)  # title + type + year + season  (all present)
    tier2: dict = defaultdict(list)  # title + type + year            (season missing)
    tier3: dict = defaultdict(list)  # title + type + season          (year missing)

    for entry in anime_list:
        title = entry.get("title")
        anime_type = entry.get("type")
        season_data = entry.get("animeSeason", {})
        season = season_data.get("season")
        year = season_data.get("year")

        has_real_year = year is not None
        has_real_season = season not in (None, "UNDEFINED")

        if has_real_year and has_real_season:
            tier1[(title, anime_type, year, season)].append(entry)
        elif has_real_year:
            tier2[(title, anime_type, year)].append(entry)
        elif has_real_season:
            tier3[(title, anime_type, season)].append(entry)
        else:
            # Both unknown — still group by all four fields for consistency
            tier1[(title, anime_type, year, season)].append(entry)

    tier1_dups = {k: v for k, v in tier1.items() if len(v) > 1}
    tier2_dups = {k: v for k, v in tier2.items() if len(v) > 1}
    tier3_dups = {k: v for k, v in tier3.items() if len(v) > 1}

    SEP = "=" * 80
    sep = "-" * 80

    # ── Internal duplicates ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  INTERNAL DUPLICATES  (same provider appears twice in one entry)")
    print(f"{SEP}")
    if not internal_duplicates:
        print("  None found.")
    else:
        print(f"  {len(internal_duplicates)} entries affected\n")
        for entry in internal_duplicates:
            from collections import Counter
            providers = [get_provider(s) for s in entry.get("sources", [])]
            duped = [p for p, c in Counter(providers).items() if c > 1]
            print(f"  {entry.get('title')}")
            print(f"    Duped providers : {', '.join(duped)}")
            for s in entry.get("sources", []):
                if get_provider(s) in duped:
                    print(f"    dup → {s}")
            print()

    def print_tier(label, dups, key_fields):
        redundant = sum(len(v) - 1 for v in dups.values())
        print(f"{SEP}")
        print(f"  {label}")
        print(f"{SEP}")
        if not dups:
            print("  None found.")
        else:
            print(f"  {len(dups)} duplicate sets  |  {redundant} redundant entries\n")
            for key, entries in dups.items():
                key_label = "  ".join(str(k) for k in key)
                print(f"  {key_label}  — {len(entries)} entries")
                for i, entry in enumerate(entries):
                    print(f"    [{i + 1}]")
                    for s in entry.get("sources", []):
                        print(f"      {get_provider(s):15} {s}")
                print()
        return redundant

    r1 = print_tier("TIER 1  (title + type + year + season)  — highest confidence", tier1_dups, 4)
    r2 = print_tier("TIER 2  (title + type + year)           — season missing",      tier2_dups, 3)
    r3 = print_tier("TIER 3  (title + type + season)         — year missing",        tier3_dups, 3)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{SEP}")
    print(f"  SUMMARY")
    print(f"{SEP}")
    print(f"  Total entries              : {total_count:,}")
    print(f"  Internal duplicates        : {len(internal_duplicates):,}")
    print(f"  Tier 1 duplicate sets      : {len(tier1_dups):,}  ({r1} redundant)")
    print(f"  Tier 2 duplicate sets      : {len(tier2_dups):,}  ({r2} redundant)")
    print(f"  Tier 3 duplicate sets      : {len(tier3_dups):,}  ({r3} redundant)")
    print(f"  Total redundant entries    : {r1 + r2 + r3:,}")
    print(f"{SEP}")


if __name__ == "__main__":
    import sys, io, contextlib, os
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/qdrant_storage/anime-offline-database-minified.json"
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"{base}_duplicates.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        find_duplicates(file_path)
    output = buf.getvalue()
    print(output, end="")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(output)
    print(f"\nReport saved to {output_path}")
