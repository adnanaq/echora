# Stage 1 — Metadata Merge Design Notes

**Date**: 2026-09-16
**Last updated**: 2026-09-21
**Status**: Design agreed, not implemented. `episode_count`, `external_sources` and the dedup approach are settled; the taxonomy rule is measured and awaiting sign-off. Company fields remain open.
**Purpose**: Capture the stage 1 investigation so the analysis is not repeated — what stage 1 must do, the per-field merge rules and the evidence for them, and what is still undecided.

---

## Table of Contents

1. [What Stage 1 Is](#what-stage-1-is)
2. [Current State](#current-state)
3. [Measured Field Behaviour](#measured-field-behaviour)
4. [Field Hierarchy](#field-hierarchy) — **the per-field rules; start here**
5. [Deduplication](#deduplication)
6. [Language-Aware Deduplication](#language-aware-deduplication)
7. [Adjacent Defects Found](#adjacent-defects-found)
8. [Implementation Plan](#implementation-plan)
9. [Company Fields](#company-fields)
10. [Taxonomy Fields](#taxonomy-fields)
11. [Open Decisions](#open-decisions)
12. [External Sources Measurement](#external-sources-measurement)
13. [Synonym Dedup Measurement](#synonym-dedup-measurement)
14. [Episode Count Measurement](#episode-count-measurement)
15. [Notes on Evidence](#notes-on-evidence)

Sections 12–14 are the evidence behind the rules in section 4; section 3 is the original day-one survey, kept for provenance.

---

## What Stage 1 Is

Stage 1 merges the seven provider records in an agent directory into one canonical `Anime`, minus the parts other stages own: episodes (stage 2), relationships (stage 3), statistics rollup (stage 4), characters (stage 5).

The key realisation: **the mappers already canonicalise**. Every `*_anime.jsonl` in an agent directory is already `Anime`-shaped — `status: "ONGOING"`, `type: "TV"`, `rating: "PG-13 - Teens 13 or older"`. Stage 1 is therefore not a parsing problem but a *reconciliation* problem: seven copies of one model that must collapse into one.

That makes `pipeline/relationship_merger.py` the template. It already solves the same shape for one field, with dual entry points (from disk via `PROVIDER_FILES`, or in-memory from `ApiFetcher`), and resolution rules of identity → sentinel suppression → priority tiebreak → union-for-arrays / first-non-null-for-scalars, with a `validate()` at the end.

---

## Current State

`scripts/process_stage1_metadata.py` is 1130 lines and predates the mapper canonicalisation. It:

- reads `kitsu.json` / `anilist.json` / `anidb.json`, when the files are `kitsu_anime.jsonl`, `anilist.jsonl`, `anidb_anime.jsonl`
- reaches into obsolete raw shapes — `mal["data"]["synopsis"]`, `kitsu["data"]["attributes"]`, `anilist["coverImage"]["extraLarge"]`
- crashes at line 1122 on `len(result.get('external_sources', {}))` when the value is `None`

The visible result is `stage1_metadata.json` with `synopsis: null`, `rating: null`, `title_japanese: null`, `genres: []` — most fields empty despite every provider supplying them.

No `libs/enrichment/src/enrichment/pipeline/metadata_merger.py` exists yet.

---

## Measured Field Behaviour

Measured across all 39 fields of the seven providers in `temp/One_agent5` (One Piece). **Only six scalars genuinely conflict.**

### Uniform — no arbitration needed

`type`, `status`, `season`, `year`, `duration`, `source_material`, `entity_type`, `aired_dates`

All agree. `aired_dates` is byte-identical across 6 of 6 providers (`1999-10-19T15:00:00Z`).

### Genuine conflicts

| field | observed |
| :---- | :------- |
| `episode_count` | anidb 1172, anime_planet 1165, offline seed 1155, stage 2 1163. Five of seven report `0` |
| `rating` | mal `PG-13`, kitsu `PG`; other five `UNKNOWN` |
| `synopsis` | seven distinct texts, all valid |
| `title` | `One Piece` ×6 vs `ONE PIECE` ×1 — casing only |
| `title_english` | same casing split |
| `title_japanese` | **romaji `ONE PIECE` ×4 vs kana `ワンピース` ×3** — semantic, not cosmetic |

### Single-source — no merge logic warranted

`background` (mal), `country_of_origin` (anilist), `licensors` (mal), `opening_themes` / `ending_themes` (mal, 30 and 27), `hiatus` (animeschedule), `titles` (anidb, 27 ISO-keyed entries)

### Union — the bulk of the work

`genres` 29 raw → 10 unique, `tags` 181 → 162, `synonyms` 32 → 23, `sources` 13 → 10, plus `external_sources`, `images`, `producers`, `studios`, `streaming_sources`, `trailers`, `themes`, `statistics`

### Correction: synopsis priority is inverted

The old script ranks AniDB first. Measured lengths:

```
anisearch 1826 > anilist 1602 > animeschedule 1184 ≈ kitsu 1181 > mal 1111 > anidb 774 > anime_planet 761
```

AniDB is the **shortest and dirtiest** — it carries a raw `http://anidb.net/ch474 [Monkey D. Luffy]` and an appended `Note:` paragraph. MAL carries `[Written by MAL Rewrite]`. AniSearch is the longest with no markup. The existing hierarchy is close to backwards.

---

## Field Hierarchy

The default tiebreak reuses `PROVIDER_PRIORITY` from `relationship_merger.py`:

```
mal > anilist > anidb > anime_planet > anisearch > animeschedule > kitsu
```

Treat that as a tiebreak of last resort, not a ranking of quality. Every field measured so far wanted a *different* winner — `episode_count` wants MAL, `synopsis` wants AniSearch, `titles` exists only on AniDB — so the order only really decides the uniform fields, where by definition it rarely matters.

### The rule that cuts across every field

**Which field a value arrives in is a provider convention, not a fact about the value.** The same concept routinely lands in different fields depending on who sent it:

| Value | One provider files it as | Another files it as |
| :---- | :----------------------- | :------------------ |
| `Shounen` | `demographics` (anilist, mal) | `genres` (animeschedule, anisearch) |
| `Super Power` | `genres` (kitsu) | `themes` (anilist) |
| Toei Animation | `studios` (anilist, animeschedule, anisearch, mal) | `producers` (anime_planet) |
| Funimation | `producers` (anilist) | `licensors` (mal) |
| Crunchyroll link | `external_sources` (anidb) | `streaming_sources` (everyone else) |
| The One Piece manga | `ADAPTATION` (anilist, mal) | `OTHER` (anime_planet, anisearch) |

So the merge must route by **what the value is**, not by the field it came in on. Wherever a rule below says "route by host" or "normalise the relation key", this is why.

### Per-field rules

Status: **settled** — decided and measured · **proposed** — measured, awaiting sign-off · **open** — needs a decision · **mechanical** — no judgement required.

| Field | Rule | Status |
| :---- | :--- | :----- |
| `episode_count` | **mal > anidb > anime_planet**. The field holds whatever the winning provider reports — no reconciliation of what the number counts. Offline seed dropped. `validate()` warns when \|stage1 − stage2\| > 10 | settled |
| `synopsis` | **Longest wins** after per-source markup stripping, not priority. Priority breaks ties within ±10% length | settled |
| `title`, `title_english` | Prefer the non-all-caps variant when normalised forms match (`One Piece` over `ONE PIECE`) | settled |
| `title_japanese` | Prefer a CJK-script value over romaji — `ワンピース` over `ONE PIECE`. Detect by Unicode range, not `langdetect`. The romaji belongs in `synonyms` | settled |
| `rating` | Sentinel-suppressed: `UNKNOWN` never wins. Only MAL and Kitsu supply a value and **they disagree** (`PG-13` vs `PG - Children`), so priority decides: mal > kitsu | settled |
| `external_sources` | `list[ExternalLink]`. Platform from the **host, not the label**; URL normalisation; route provider/streaming/trailer links away | settled |
| `synonyms` | Union, then the deterministic ladder — NFKC, apostrophe and alphanumeric folding, plus romaji long-vowel folding. **No embedding model** | settled |
| `images` | Plain union. Dedupe only on identical URL — no priority ordering, and multiple images from one provider are all kept. Note MAL alone supplies 20 covers | settled |
| `tags` | Full union, nothing dropped — and the destination for any value that matches no taxonomy vocabulary. See [Taxonomy Fields](#taxonomy-fields) | settled |
| `studios`, `producers`, `licensors` | Undecided — see [Company Fields](#company-fields) | **open** |
| `genres` | Union, then **classified against the MAL ∪ AniList genre vocabulary** (21 terms; the two agree exactly). A value outside it is not a genre. See [Taxonomy Fields](#taxonomy-fields) | proposed |
| `themes` | Union, then classified: MAL's 52 themes plus AniList's `Theme-*` tags. Everything left over goes to `tags` | proposed |
| `demographics` | Union, then classified against the agreed five (Josei, Kids, Seinen, Shoujo, Shounen). Reclaimed from other providers' `genres` — AnimeSchedule files `Shounen` there | proposed |
| `related_source_material` | Union keyed by relation, but the keys disagree (`ADAPTATION` vs `OTHER`) — needs the same normalisation stage 3 applies via `_pick_relation` | **open** |
| `statistics` | Provider-keyed dict, plain merge. Apply `normalize_score` — AniSearch reports on a /5 scale (`4.18`), everyone else /10. Metric sets differ per provider and that is fine | mechanical |
| `aired_dates`, `broadcast` | **Sub-field merge, never whole-object.** Providers partition these: mal/anisearch carry `day`/`time`/`timezone`, anilist/kitsu carry `next_episode_at`, animeschedule carries `jp_time`/`sub_time`/`dub_time`/`premiere_*` | mechanical |
| `sources` | Union of provider URLs. AnimeSchedule alone supplies six cross-provider links, which makes it a useful secondary identity spine | mechanical |
| `streaming_sources` | Union, deduped by platform + normalised URL | mechanical |
| `trailers` | Union by normalised URL. Kitsu and MAL supply *different* videos for the same anime, so both are kept; MAL also carries `title` and `thumbnail` | mechanical |
| `type`, `status`, `season`, `year`, `duration`, `source_material`, `entity_type` | Uniform across every provider that supplies them. First non-null by priority; **log on disagreement**, since divergence means a source bug | mechanical |
| `nsfw` | AniList and Kitsu both supply it and agree (`False`). First non-null | mechanical |
| `background` (mal), `opening_themes` / `ending_themes` (mal, 30 and 27), `licensors` (mal), `country_of_origin` (anilist), `hiatus` / `month` (animeschedule), `titles` (anidb, 27 ISO-keyed) | Single-source passthrough — nobody to arbitrate with | mechanical |
| `content_warnings` | No provider supplied any in the sample. Union when present | mechanical |
| `related_anime` | Stage 3 | not stage 1 |
| `id`, `similarity_score`, `score`, `staff_data` | System-generated or computed; `staff_data` is not populated by any provider at anime level | not stage 1 |

Coverage note: **AniDB and Kitsu supply no company data at all**, and AniDB supplies neither `genres` nor `duration`. AniDB is the only source of `titles`, and carries by far the most `tags` (122 against AniList's 34).

---

## Deduplication

Four tiers, cheapest first.

**Tier 1 — normalisation.** `normalize_string_for_comparison` is currently only `.lower().strip()`. Measured on the real synonym pool: 92 raw → 34 with today's normalisation → **32** with NFKC plus apostrophe folding. That last step collapses `All'arrembaggio!` / ``All`arrembaggio!`` / `All’arrembaggio!` (U+0027 / U+0060 / U+2019). Deterministic, no model required.

**Tier 2 — title bleed.** `One Piece` appears in AniDB's *synonyms*, and case-folding kept `ONE PIECE` while discarding the correctly-cased primary. Synonyms must be filtered against the resolved `title` / `title_english` / `title_japanese`, which means synonym dedup runs **late** in the merge, not early.

**Tier 3 — `external_sources` key canonicalisation.** See [Adjacent Defects](#adjacent-defects-found).

**Tier 4 — semantic.** Requires an injected embedding model; see below.

---

## Language-Aware Deduplication

**`deduplicate_synonyms_language_aware()` silently no-ops today.** Both it and `deduplicate_semantic_array_field()` fall back to `deduplicate_simple_array_field(values, [])` when `embedding_model is None` (`deduplication.py:207-211` and `:124-127`), and stage 1 injects no model. Its own docstring example — collapsing the duplicate Italian variant — does not happen in the current pipeline. It degrades the same way when `langdetect` is unavailable.

Two findings that change how it should be used:

**Use `titles`, not `langdetect`, where available.** AniDB ships `titles` as an ISO-keyed map (`de`, `ru`, `ko`, `th`, `tr`, `zh-Hant`, `zh-Hans`, …). That is ground truth. Asking `langdetect` for the language of the two-character string `海贼`, or of the token `optv`, is unreliable by construction — short strings are its documented weak case, and a misgroup silently drops a real variant. Seed language groups from `titles`; fall back to `langdetect` only for synonyms with no ISO-coded source.

**There is a genuine win case.** Within the Greek group, `Ντρέηκ, το Κυνήγι του Θησαυρού` and `Ντρέικ και το Κυνήγι του Θησαυρού` differ by one letter plus a conjunction — string dedup keeps both, semantic dedup at 0.85 collapses them. Correctly *not* collapsed: `海贼` ("sea thief") and `海贼王` ("sea thief king") are distinct titles despite high lexical overlap, which is exactly why the within-language constraint matters.

Conclusion: worth wiring, but only with a model injected, and as the **last** tier after normalisation has already done the free 92 → 32.

---

## Adjacent Defects Found

**Stage 4 is dropping data that is already shaped correctly.** Every provider emits `statistics` as `{"<provider>": {...}}` — merging is `dict.update()` seven times. Yet `stage4_statistics.json` is `{"statistics": {}}`. Same obsolete-shape defect as stage 1. Separately, `normalize_score` exists in `text_utils.py` and is not applied, so anisearch's `4.18` (a /5 scale) is merged alongside /10 scores.

**`external_sources` keys are unnormalised.** 33 keys across providers resolve to 31 URLs across 22 hosts — roughly 18 real entities:

- `YouTube` / `Youtube` / `youtube` — three keys, three *different* URLs (channel, `@handle`, a single video)
- `Official Site` / `official site` / `official` / `official_website` / `One Piece` / `Toei Animation` — six keys, three distinct URLs
- `Syoboi`/`syoboi`, `Wikipedia`/`wikipedia_jp`, `Instagram`/`instagram`, `ANN`/`anime_news_network`

The field is `dict[str, str]`, which structurally **cannot hold two YouTube links**. Stage 1's merge rule compounds it: `if key not in external_sources` means first-provider-wins, so an arbitrary key either wins or loses on iteration order.

Also measured: `external_sources` overlaps `sources` on 2 of 31 entries, and overlaps `streaming_sources` on 4 hosts (crunchyroll, netflix, hulu, youtube) because AniDB files streaming platforms there while everyone else uses `streaming_sources`. The agreed rule is that **`external_sources` is the residual** — what remains after `sources`, `streaming_sources` and `trailers` have claimed their own.

> Superseded in part — see [External Sources Measurement](#external-sources-measurement) for figures measured across 320 anime rather than One Piece alone. The key-normalisation and field-shape problems remain.
>
> The AniDB half has since changed substantially. Correcting the resource-type map raised AniDB's contribution by 33% (492 → 652 entries across 101 anime, averaging 6.5 keys per anime) and its keys are now already canonical, so the alias work applies only to the other six providers. Routing is also larger than One Piece suggested: measured across 100 anime, 20% of AniDB's entries are provider pages belonging in `sources` and 8% are streaming platforms, leaving 71% as true residual.

---

## Implementation Plan

| Phase | Work |
| :---- | :--- |
| 0 | Freeze `temp/One_agent5` as a fixture; capture current broken output as a baseline |
| 1 | `pipeline/metadata_merger.py` — dual-mode, mirroring `relationship_merger.py`. Reuse `PROVIDER_PRIORITY`, `PROVIDER_FILES`, `_SENTINELS`, `is_signal` |
| 2 | Classes A / B / D / E — the mechanical 80%, no model needed |
| 3 | Class C overrides and Class F sub-field merge for `broadcast` / `aired_dates` |
| 4 | Dedup: NFKC normaliser, `external_sources` key canonicalisation, title-bleed filter |
| 5 | Optional embedding-model injection for language-aware synonym dedup, seeded from `titles`. Must fail loudly, not silently, when absent |
| 6 | `validate()` — episode_count drift, cross-source ID disagreement, all-sources-UNKNOWN fields |
| 7 | `process_stage1_metadata.py` → thin CLI, 1130 lines → ~60 |
| 8 | Tests; then apply the statistics finding to stage 4 separately |

Each phase stays within the 500-line file / 50-line function limits; the merger should land around 400 lines.

---

## Company Fields

`studios`, `producers` and `licensors` are the one group still undecided. Two separate problems sit underneath them.

### Problem 1 — role is a property of the work, not the company

Measured across the offline database (40,346 entries): **1,219 company names (18%) appear as both a studio and a producer**, and for the large ones both are substantial.

| Company | studio credits | producer credits |
| :------ | -------------: | ---------------: |
| toei animation | 391 | 1047 |
| sunrise | 230 | 587 |
| production i.g | 255 | 454 |
| tms entertainment | 192 | 409 |

Toei genuinely animates some titles and produces others. So a company cannot be classified once and for all — the role belongs to the *credit*, not the company. That settles the semantic question: these fields describe what a company did **on this anime**, which is MAL's taxonomy.

This has an awkward consequence for AniList. Our mapper splits its single studio list on `isAnimationStudio` (`anilist_mapper.py:176-187`) — a *company* property — to populate per-work role fields. AniList also exposes `isMain`, the actual per-work signal, and the query fetches it (`anilist_helper.py:312-320`) but the mapper never reads it. AniList has no concept of a licensor at all, which is why Funimation and 4Kids land under `producers` there and `licensors` on MAL.

### Problem 2 — the same company arrives under different names

Raw provider records for One Piece, all seven providers:

```
anilist        Toei Animation             anilist.co/studio/18
animeschedule  Toei Animation             animeschedule.net/studios/toei-animation
anisearch      Toei Animation Co., Ltd.   anisearch.com/company/412,toei-animation-co-ltd
mal            Toei Animation             myanimelist.net/.../producer/18/Toei_Animation
anime_planet   Toei Animation             (no url)
```

Each provider supplies its own company URL, so there is **no shared identity key** — a folded name is the only available join.

At corpus scale the variation is large. Two folds over all 6,786 company names:

| Fold | Names merged | Credits affected |
| :--- | -----------: | ---------------: |
| punctuation + spacing only | 201 | 6,765 |
| **+ legal suffixes** (`Co., Ltd.` / `Inc.` / `K.K.` / `Corp.` / `LLC`) | **703** | **38,381** |

```
2175 credits  toei animation  |  toei animation co., ltd.
1289 credits  sunrise  |  sunrise co., ltd.  |  sunrise inc.
1220 credits  j.c. staff  |  j.c.staff  |  j.c.staff co., ltd.
 851 credits  madhouse  |  madhouse inc.  |  mad house, inc.
```

**Stop at legal suffixes.** An earlier attempt also stripped industry words — `studio`, `production`, `entertainment` — and that merged `tsuburaya entertainment` with `tsuburaya productions`, which are distinct entities, and `studio anima` with `anima`. Legal suffixes are never part of an identity; industry words often are.

### Correction to an earlier claim

An earlier check of One Piece concluded that provider company names match exactly, and that name normalisation was therefore unnecessary. That check covered only three providers and missed AniSearch, which writes `Toei Animation Co., Ltd.` The variation appears on the very first title in the raw merge inputs; the claim was wrong.

### What is still open

The fold is straightforward — punctuation, spacing and legal suffixes, the same deterministic shape validated for synonyms. What is not settled is the shape, because role is per-work and providers disagree on granularity:

- **Union per field** — keep every provider's placement. Funimation lands in both `producers` and `licensors`; nothing is lost and no provider is overruled, but the same company appears two or three times across the record.
- **One provider decides the role** — MAL has the finest taxonomy. Each company lands in exactly one field, at the cost of discarding other providers' placement and losing companies MAL does not list.
- **A single `companies` list, role as a field** — each company appears once carrying the roles it was credited with, sources accumulated. Loses nothing and matches the per-work finding, but it is a model *and* proto change, the same commitment as `external_sources`.

### Limits of this measurement

The corpus figures come from the offline database, which is already merged and lowercased — raw provider data will be at least this messy, and carries case variation the corpus does not. The raw cross-provider comparison is One Piece only; it should be repeated across a spread of titles before the fold is relied upon.

---

## Taxonomy Fields

`genres`, `themes`, `demographics` and `tags` overlap: the same value arrives in different fields depending on the provider. **Rule: a value belongs to exactly one of them.**

Measured 2026-09-21 over 60 anime spread across eras (AniList and Kitsu, both APIs) plus MAL's published vocabulary. The One Piece figures this replaces came from a title with no MAL Themes row at all, which made MAL's vocabulary look far less useful than it is.

### Genres and demographics are not ambiguous

**All 17 AniList genres in the sample are in MAL's genre vocabulary — zero unknown.** Both providers also use the identical five demographics. So two of the three fields have a settled cross-provider vocabulary:

- **genres** (21): Action, Adventure, Avant Garde, Award Winning, Boys Love, Comedy, Drama, Fantasy, Girls Love, Gourmet, Horror, Mystery, Romance, Sci-Fi, Slice of Life, Sports, Supernatural, Suspense, plus the explicit Ecchi, Erotica, Hentai
- **demographics** (5): Josei, Kids, Seinen, Shoujo, Shounen

A value a provider files under `genres` that is not in that list is not a genre. That is what corrects AnimeSchedule's `Shounen` and AniSearch's `Fighting-Shounen` and `Ganbatte`.

### The disagreement is entirely theme-versus-tag

AniList carries a category on every tag. Across the sample its 154 distinct tags split:

```
Theme 83    Cast 42    Setting 15    Technical 8    Demographic 5
```

MAL knows 11 of the 83 `Theme-*` tags but 14 of the 70 `Cast`/`Setting`/`Technical` ones — MAL files `Adult Cast`, `School`, `Space` and `Military` as themes where AniList calls them Cast or Setting. The two taxonomies disagree far more on the theme/tag boundary than on genre/theme.

### The rule

Classify each value once, in this order; it lands in exactly one field.

| Field | Source of truth | Approx. size |
| :---- | :-------------- | -----------: |
| `demographics` | the agreed five | 5 |
| `genres` | MAL ∪ AniList genre vocabulary (identical) | 21 |
| `themes` | MAL's 52 themes plus AniList's `Theme-*` tags | ~130 |
| `tags` | everything else — AniList `Cast`/`Setting`/`Technical`, Kitsu leftovers, AniSearch's own labels | open-ended |

Two things make this cheap. The authorities already exist — nothing has to be invented or hand-maintained beyond MAL's published lists. And the pipeline already does it for one provider: `anilist_mapper.py:165-174` routes AniList tags by their category today. The change is applying that classification to **every** provider's values rather than trusting the field they arrived in.

### The four steps

1. **Fold case and spacing** so `Shounen` and `shounen` are one value. Some providers lowercase everything; without this the same word survives twice.
2. **Look the value up** in the lists.
3. **Promote it as far as it goes**, stopping at the first match: `demographics > genres > themes > tags`.
4. **Store the list's spelling**, not the provider's.

Step 4 matters as much as the others. Without it the merge produces `Shounen` *and* `shounen` in `demographics`, and `Action` *and* `action` in `genres` — the classification is right but the duplication survives. Adding it takes One Piece from 225 values to 195.

### The precedence order, and why

The hierarchy compares **lists**, never where a provider filed the value. Provider placement is ignored entirely.

| Rank | Field | Why it sits there |
| :--- | :---- | :---------------- |
| 1 | `demographics` | Smallest and most specific list — 5 words. A demographic should never be allowed to rest as a genre |
| 2 | `genres` | Highest confidence: MAL and AniList agree 17/17 on what a genre is |
| 3 | `themes` | Larger and fuzzier — MAL's 52 plus AniList's labelled ones — so it yields to the two above |
| 4 | `tags` | The catch-all |

Worked examples, with the provider disagreement deliberately disregarded:

| Value | Providers filed it as | On which list | Resolves to |
| :---- | :-------------------- | :------------ | :---------- |
| Shounen | demographic (AniList, MAL), genre (AnimeSchedule), theme (Kitsu) | MAL demographic | demographics |
| Super Power | genre (Kitsu), theme (AniList) | MAL theme | themes |
| Action | genre ×5, theme (Kitsu) | MAL genre | genres |

### How often two lists actually disagree

Measured: MAL's three lists have **zero overlap with each other**, and no AniList theme collides with a MAL genre or demographic. So genre-versus-theme and genre-versus-demographic conflicts do not occur.

One conflict exists — 9 values MAL calls themes that AniList files as `Cast` or `Setting`: `crossdressing`, `delinquents`, `detective`, `historical`, `samurai`, `school`, `space`, `urban fantasy`, `villainess`. AniList's `Cast`/`Setting` maps to `tags` here, so the order promotes them to `themes`, which is the better reading.

The precedence therefore does real work in exactly one place today, but it is deterministic if either site later introduces a value that lands on two lists.

### Nothing is lost

Verified over all seven providers for One Piece:

```
272 raw entries  ->  195 distinct after folding  ->  195 placed  ->  0 lost
   demographics   1
   genres         6
   themes        45
   tags         143
```

The 77 that vanish between 272 and 195 are the same word arriving from several providers — collapsed, not dropped.

### Who actually decides

The rule borrows two sites' editorial judgement rather than exercising its own:

| Decider | Values | Effect |
| :------ | -----: | :----- |
| MAL's published lists | 14 | genres and demographics |
| AniList's own labels | 38 | themes |
| The default | 11 | values nobody classified that were not already tags |
| Nothing | 132 | already tags, stay tags |

So the default only *moves* 11 values. The other 132 are mostly AniDB's tags, which never claimed to be anything else. The 11 that move are `fighting-shounen` and `ganbatte` (AniSearch filed as genres), `friendship` (Kitsu had it in all three), and eight Kitsu themes that were already tags as well.

Three consequences worth stating plainly:

- **The lists are a snapshot.** If MAL adds a genre, it lands in `tags` until someone refreshes the list. That has to be a deliberate occasional chore, not silent drift.
- **Four providers get no say.** AniSearch, Anime-Planet, AnimeSchedule and AniDB carry no category information, so nothing unique to them can ever be promoted above `tags`. AniDB's 122 tags are the bulk of the data and stay tags by construction.
- **AniDB's tags contain junk.** `maintenance tags` is in there — a note about the database, not the anime. A separate data-quality problem this rule does not address.

It also fixes Kitsu without special-casing. Kitsu's flat `categories` list has no genre/theme distinction, so it emits the same value into both — on One Piece, `action`, `adventure`, `comedy`, `fantasy`, `friendship` and `super power` all appear as both `kitsu.genres` and `kitsu.themes`. Classifying rather than trusting removes the duplication.

### Effect on the One Piece conflicts

Seven of 57 values appeared in more than one field. Under the rule:

| Value | Arrived as | Resolves to |
| :---- | :--------- | :---------- |
| Action, Adventure, Comedy, Fantasy | genres + `kitsu.themes` | **genres** |
| Super Power | `kitsu.genres` + themes | **themes** |
| Martial Arts | themes | themes |
| Shounen | demographics + `animeschedule.genres` + `kitsu.themes` | **demographics** |
| Friendship | `kitsu.genres` + `kitsu.themes` | **tags** — in no vocabulary and no AniList category |

### Limits of this measurement

60 anime, AniList and Kitsu only. MAL's side rests on its published vocabulary rather than crawled data — how often MAL actually populates its Themes row is unmeasured, and One Piece (which has none) is the only MAL taxonomy sample taken. AniSearch and AnimeSchedule were not sampled beyond One Piece; both supply flat genre lists with no category information, so they will always be classified rather than trusted.

---

## Open Decisions

**1. `episode_count` source.** Resolved 2026-09-17 — see [Episode Count Measurement](#episode-count-measurement). **Hierarchy: mal > anidb > anime_planet**, with `validate()` warning when `|stage1 − stage2| > 10`. The offline seed is dropped as a source entirely: its upstream is archived, so it can only grow more stale.

The field holds whatever the winning provider reports; the merge does not reconcile what the number counts. Note that MAL's number is the *announced total* for a cour show and the *aired count* only for open-ended ones — see the correction in that section.

**2. `external_sources` shape.** **Decided 2026-09-19: `list[ExternalLink]`** (`platform`, `source`, `label`, `language`) — see [External Sources Measurement](#external-sources-measurement). A mapping loses 24% of external links on popular titles, and those survive the full dedup ladder, so they are distinct pages and accounts rather than spelling or URL variants. It mirrors the existing `StreamingEntry` / `TrailerEntry` / `CompanyEntry` pattern. Not yet implemented: it is a model **and proto** change touching five mappers, plus proto regeneration and a reindex.

**3. Embedding model in stage 1.** **Closed 2026-09-17: do not wire it** — see [Synonym Dedup Measurement](#synonym-dedup-measurement). The deterministic ladder is better on accuracy *and* recall at every threshold tested, so this was never a cost trade-off. Nothing is blocked: no model is injected today, so this is already the state. The two follow-ups — romaji long-vowel folding in Tier 1, and making the silent no-op fail loudly — land inside the phase 4/5 rewrite.

**4. Company fields.** Open — see [Company Fields](#company-fields). The name fold is straightforward (punctuation, spacing, legal suffixes). What is undecided is the shape, because role is a property of the work and providers disagree on granularity: union per field, one provider decides, or a single `companies` list carrying roles.

**5. `genres` / `themes` / `demographics` overlap.** Measured 2026-09-21 — see [Taxonomy Fields](#taxonomy-fields). **Proposed: classify every value against a vocabulary instead of trusting the field it arrived in**, so each value lands in exactly one of demographics / genres / themes / tags. Genres and demographics turn out to be unambiguous (AniList and MAL agree 17/17 and 5/5); only the theme/tag boundary is contested. Awaiting sign-off.

---

## External Sources Measurement

Measured 2026-09-17, after the AniDB resource-type fix. Replaces the earlier One Piece–only figures.

**Method.** Collect `(anime, label, url)` from each provider, then run the cleanup any design would do first — platform derived from the host, URLs normalised, provider/streaming/trailer links routed to the fields that already exist for them — and count what remains: *one platform holding two different links for one anime*. That residual is the only thing a mapping cannot represent, so it is the number the shape decision rests on.

**Sample.** AniList 300 most popular + 145 random (API), MAL 22 and AniSearch 20 of those same popular titles (browser).

### Loss under `dict[str, str]`

| Source | anime | colliding slots | links lost | per anime |
| :----- | ----: | --------------: | ---------: | --------: |
| AniList, random titles | 145 | 5.7% | 8 | 0.09 |
| AniList, popular titles | 300 | 14.0% | 121 | 0.40 |
| MAL, popular | 22 | 3.3% | 3 | 0.14 |
| AniSearch, popular | 20 | 17.9% | 5 | 0.25 |
| **All three merged** | **20** | **20%** | **29 (24% of all links)** | **1.45** |

Two things this shows:

**Popularity drives it.** Obscure titles carry two or three links and rarely collide; popular ones carry many. A sample of random titles understates the problem roughly four-fold.

**Merging multiplies it.** Each provider alone looks tolerable (3–18%). Merged, the rate more than doubles, because providers hold *different* accounts for the same platform — MAL has `@heroaca_anime`, AniList has `@MHAOfficial`, and both are real.

### What the collisions actually are

Almost all are language or region variants of one thing:

```
Chainsaw Man, twitter   CHAINSAWMAN_PR  Chainsaw_EN  Chainsaw_FR
                        ChainsawMan_PT  chainsawman_es  chainsawman_la
Demon Slayer, twitter   kimetsu_off (JP)  DemonSlayerUSA  kimetsu_fr
The Promised Neverland  neverland-anime.com  neverland-anime.com/1st
                        neverland-animeusa.com/1st
```

AniList tags each link with a language, which separates **81%** of collisions cleanly. It is not sufficient on its own: 4% are two links sharing one language (Chainsaw Man has both `_es` and `_la` Spanish accounts; Death Parade has two Japanese official sites, `ntv.co.jp` and `vap.co.jp`), and 15% have no language at all. So a `(platform, language)` key is better than `platform` but still lossy — and it depends on metadata only AniList supplies.

### Providers key by different things

This is why the label cannot be the platform name:

| Provider | What the key actually is | Example |
| :------- | :----------------------- | :------ |
| AniList | generic category | `Official Site`, `Twitter` |
| MAL | the account handle | `@MHAOfficial`, `@heroaca_anime` |
| AniSearch | the page's own title | `Toei Animation`, `Kimetsu`, `Gkids` |
| AniDB | the platform | `twitter`, `allcinema` (already canonical) |

AniSearch's and MAL's keys are per-link titles, not platform names — so the "33 keys" counted earlier for One Piece were never 33 platforms. Those titles carry real information (`Toei Animation` vs `One Piece` distinguishes the studio's page from the show's), and a platform-keyed mapping has nowhere to put it. `ExternalLink.label` does.

### The dedup ladder, and what it deliberately leaves alone

The canonicalisation ladder is the same either way, and it runs *before* the 24% is counted:

| Step | Catches | Example |
| :--- | :------ | :------ |
| Platform from the **host**, never the label | every spelling variant, with no alias list to maintain | `YouTube` / `Youtube` / `Official Site` / `Ani-One Asia` → `youtube` |
| Normalise the URL | scheme, `www.`, trailing slash, percent-encoding | `https://one-piece.com/` = `https://one-piece.com` |
| Fold host aliases | renamed services | `x.com/foo` = `twitter.com/foo` |
| Strip tracking parameters | share links | `twitter.com/anime_shingeki?t=04jz…&s=09` = `twitter.com/anime_shingeki` |
| Route by host | links owned by another field | provider pages → `sources`, platforms → `streaming_sources`, `watch?v=` → `trailers` |

What it deliberately does **not** collapse:

- **Different paths on one host** — `shingeki.tv` and `shingeki.tv/season1` are different pages; `twitter.com/kimetsu_off` and `twitter.com/DemonSlayerUSA` are different accounts.
- **YouTube `/@handle` vs `/channel/<id>`** — confirmed manually to be the same channel for One Piece (`@onepieceofficial` = `channel/UCdAHaWcKdpbT5XkN2Er6BUQ`). Proving it in general needs a network lookup, and it occurs once in 487 anime; folding it moves the loss from 24% to 23%, so the dependency is not worth taking. Under a list the failure mode is a visible duplicate entry; under a mapping it is silent loss of a real link.

Note that One Piece still has **two** genuine YouTube channels once that pair is folded: `@onepieceofficial` (Japanese) and `@OnePieceOfficialENG` (English) — the same language-variant pattern as Twitter.

### Routing must not depend on which provider spoke

The same URL can be classified differently by different providers. AniList tags both One Piece YouTube channels `type: STREAMING` (episodes stream there), while MAL supplies `@onepieceofficial` with no type at all — so one copy routes to `streaming_sources` and the other to `external_sources`.

Measured: of 57 links supplied by more than one provider, **1 (2%)** routed inconsistently. Rare, but it makes output depend on provider order, which is the same defect as the first-key-wins merge rule.

Rule: a link is streaming if its host is a known streaming platform **or any** provider marked it `STREAMING`. Union the providers' knowledge rather than letting the last writer win.


### AniDB does not force the change

After the type-mapping fix and routing, AniDB yields at most one URL per platform: its genuine multi-value resources are either trailers (161 of 164 type-26 identifiers are `watch?v=` videos, which belong in `trailers`) or several *different works* that are correctly skipped. The case for a list rests on the other providers.

### Correction to an earlier figure

An earlier note claimed a mapping "must discard roughly 6 real links" for One Piece. That count was taken before URL normalisation and routing; most of those six were the same channel written two ways, or trailers. Measured properly the One Piece residual is two (Twitter accounts, and the Toei studio page vs the show site). The 24% figure above is the trustworthy one because it is measured after cleanup and across 20 titles.

---

## Synonym Dedup Measurement

Measured 2026-09-17 over 393 synonym values from 40 anime in the offline database (seed 23, titles with ≥6 synonyms). Compares the deterministic normalisation ladder against `deduplicate_semantic_array_field` using BGE-M3.

Every value the model removed that plain normalisation kept was classified by hand against its closest surviving match. A removal is wrong when the two values denote different things — a different season, a different release format, or another locale's title.

| Approach | Removed | Wrong | Breakdown |
| :------- | ------: | ----: | :-------- |
| **deterministic ladder** | **88** | **0** | — |
| semantic @ 0.80 | 160 | 32 | 17 season, 10 script, 5 format |
| semantic @ 0.85 (shipped default) | 125 | 14 | 5 season, 5 script, 4 format |
| semantic @ 0.90 | 90 | 6 | 4 format, 1 script, 1 season |
| semantic @ 0.95 | 51 | 1 | 1 season |
| semantic @ 0.98 | 26 | 0 | — |

The model loses on both axes at once. At equal recall (0.90, ~90 removals) it makes 6 wrong merges against 0. At equal precision (0.98, no wrong merges) it removes 26 against 88 — 3.4x fewer. No threshold is ahead on both.

Real examples of the wrong merges:

```
Mask Danshi…        dropped 'マスク男子は恋したくないのに OAD'   the OAD is a separate release
Minky Momo          dropped 'مغامرات حنين 2'                   season 2, while season 1 was kept
The New Gate        dropped 'ザ・ニュー・ゲート'                 the entire Japanese title
Renzu               dropped 'Рензу: Растояние между двумя'     the only Russian title
Jing Cui Xian Zun   dropped 'Jing Cui Xian Zun: Part II'       Part 2 kept, Part II dropped
```

It also dropped `The Legend of Heroes: Trails in The Sky` — the official English title — in favour of the romaji. For a search index that is the most valuable synonym in the list.

**Why it fails is structural, not a tuning problem.** Embeddings encode meaning, and "X Season 1" and "X Season 2" mean nearly the same thing: the distinguishing token is a single digit inside a 1024-dimensional vector. Anime synonyms are precisely the case where small differences carry the information — season and part numbers, `OAD`/`OVA`/`Movie` markers, script variants. String folding never touches those, because digits and markers survive it untouched.

### The one win worth taking

Most of what the model legitimately caught is transliteration noise: `Tēkyū` vs `Teekyuu`, `Mahō` vs `Mahou`. Japanese long vowels are written either with a macron or by doubling, so folding both to one form is a deterministic rule — strip diacritics, then collapse `ou`/`oo` → `o` and doubled vowels to single. That is what takes the ladder from 72 removals to 88, at 0.03 ms per anime, and it cannot merge two different works.

Add it to Tier 1 rather than adding a tier.

### Efficiency, for completeness

```
deterministic ladder :   1.3 ms total   (0.03 ms per anime)
model load (one off) :   1.3 s
semantic dedup       :   4.5 s total    (113 ms per anime)
```

~3,500x slower excluding model load, plus a 6.4 GB model. Across 40,346 anime that is roughly 1.3 seconds against 75 minutes.

### Limits of this measurement

40 anime, 393 values, one seed. The wrong-merge classification is a hand judgement, not independent ground truth. The comparison used `deduplicate_semantic_array_field`, which does no language grouping; `deduplicate_synonyms_language_aware` would remove the SCRIPT class — 1 of the 6 errors at threshold 0.90, leaving 5 against 0.

Note also that `_is_semantically_duplicate` binds its threshold as a *default argument*, fixed at import time. Reassigning `SEMANTIC_SIMILARITY_THRESHOLD` at runtime has no effect — worth knowing before anyone tries to tune it from config.

---

## Episode Count Measurement

**Decision: `mal > anidb > anime_planet`, and the field holds whatever the winning provider reports.** The merge does not try to reconcile what the number counts.

That last point matters, because the number does not mean the same thing everywhere.

### Correction — "MAL counts aired episodes only" was wrong

An earlier version of this section justified putting MAL first by claiming it counts only aired episodes. That is true for open-ended shows and false for everything else. Measured across four cases on 2026-09-19:

| Show | MAL anime page | Episode-list counter | Stored | Which quantity |
| :--- | :------------- | :------------------- | -----: | :------------- |
| One Piece (open-ended) | `Unknown` → 0 | `(1174/Unknown)` | 1174 | **aired** |
| Tsuihou Sareta (airing cour) | 26 | `(12/26)` | 26 | **announced total** |
| Seitokai ni mo Ana wa Aru! (not started) | 12 | `(0/12)` | 12 | **announced total**, nothing aired |
| Aoashi 2nd Season (not started) | 0 | `(0/Unknown)` | 0 | nothing known |

The episode-list fallback only fires when the anime page is empty, so the aired-only path is reached solely for shows with no announced total. For a cour show we store the forward-looking number — including, for Seitokai, a count of 12 for a show where nothing has aired yet.

### What each provider reports, for One Piece

| Source | Count | Note |
| :----- | ----: | :--- |
| MAL episode-list counter (live) | 1174 | reached only because One Piece has no announced total |
| AniDB `<episodecount>` (live) | 1184 | 1178 aired, 2 scheduled ahead, 4 undated |
| AniSearch (June snapshot) | 1200 | already past September's aired count, in June |
| Kitsu (June snapshot) | 1388 | 214 beyond September's aired count |
| Anime-Planet (June snapshot) | 1165 | |
| stage 2 (June snapshot) | 1163 | |
| offline seed (2026-01-10) | 1155 | frozen — upstream archived |

Kitsu's June figure exceeding September's actual count is the clearest overcount. The seed is dropped regardless: its upstream is archived, so it can only grow staler.

If an aired-episode count is ever wanted, it does not need a provider — stage 2 emits episode records carrying `aired`, so counting those with a past date gives it directly.

### Why MAL looked unusable, and was not

MAL renders `Episodes: Unknown` on the anime page for any long-running show without a planned total, so `episode_count` maps to 0 — which is what the original field-behaviour table recorded. But `mal_episode_count_crawler` already reads the episode list page counter (`(1,174/Unknown)`) and takes the number before the slash, and `MalHelper._fetch_anime` already called it as a fallback.

The reason the value never reached stage 1 was an ordering bug: `fetch_mal_anime(url, output_path=...)` persisted the record, and only then was `episode_count` patched on the returned dict. The caller saw 1174; the file kept 0; stage 1 reads the file. Verified live before the fix — `returned in memory: 1174`, `written to jsonl: 0`.

Fixed by resolving the count before persisting. The fallback design is unchanged: a show with an announced total gets its count from the anime page for free, and the extra page fetch happens only when that is empty.

### A second bug, found the same way

MAL accepts a bare `/anime/{id}`, but its episode and character pages hang off the full address with the title segment. Glued onto a bare url, `/episode` is read by MAL as the title, so the sub-page fetch silently returns nothing.

`fetch_all` guarded against that by rejecting bare urls outright — and **29,863 of the 29,864 MAL urls in the seed are bare**. The single exception is One Piece, whose entry had been edited by hand locally; upstream (both manami's final release and the successor) has none. So MAL, the top-priority provider for this field, was being skipped for every anime except the one used for testing.

The page names its own full address in its canonical link, so the crawler now reads it there, returns it as the anime's source, and the helper uses it for the episode and character pages. Verified live from bare seed-format urls: One Piece resolved to 1174 (previously 0), and a three-episode OVA fetched 3 episodes and 17 characters where it would previously have returned nothing.

### Seed database status

`manami-project/anime-offline-database` and its generator `modb-app` were both archived in early July 2026, along with the rest of that author's project. The final release is `2026-27` (41,537 entries, 2026-07-04) and remains downloadable, since archived repositories keep their release assets.

An active continuation exists: [`cedya77/anime-offline-database`](https://github.com/cedya77/anime-offline-database), weekly releases, schema identical to ours, and carrying more provider IDs than manami's final on every source (it merges duplicate entries, so the entry count is lower while ID coverage is higher). [Fribb/anime-lists](https://github.com/Fribb/anime-lists) already sources from it.

This does not change the hierarchy above — the seed was the weakest source even before it froze — but it does mean the seed can be refreshed rather than abandoned.

---

## Notes on Evidence

Field-behaviour and statistics figures are measured against the agent directories, of which `One_agent2` through `One_agent5` each hold all seven providers. All four produce identical output, so that is **one work (One Piece) captured four times, not four samples** — enough to design against, not enough to generalise. Those should be re-checked once a second title has a complete run.

Three areas have since been measured more widely and their figures supersede the One Piece ones:

| Area | Sample |
| :--- | :----- |
| [External Sources](#external-sources-measurement) | 320 anime — AniList 445, MAL 22, AniSearch 20, AniDB 133 |
| [Synonym dedup](#synonym-dedup-measurement) | 393 values from 40 anime in the offline database |
| [Taxonomy fields](#taxonomy-fields) | 60 anime across three eras, AniList and Kitsu |

[Company fields](#company-fields) sit in between: the role and name-variation counts come from the offline database's 40,346 entries, but the raw cross-provider comparison is still One Piece alone.

One earlier note here was wrong and is worth flagging: it predicted a regenerated fixture would show **fewer** AniDB keys after ECHO-46. The opposite happened — correcting the resource-type map raised AniDB's contribution by 33%.
