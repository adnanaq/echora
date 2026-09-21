# Stage 1 — Metadata Merge

**Last updated**: 2026-09-21
**Purpose**: How seven provider records become one canonical `Anime`, what rule governs each field, and the evidence behind the rules that are not obvious.

Read [Field Hierarchy](#field-hierarchy) first — it is the contract. The sections after it exist so that nobody has to re-measure what has already been measured, and so that a rule that looks arbitrary is not "simplified" back into a bug.

---

## Table of Contents

1. [What Stage 1 Is](#what-stage-1-is)
2. [Where the Code Lives](#where-the-code-lives)
3. [How the Merge Works](#how-the-merge-works)
4. [Field Hierarchy](#field-hierarchy) — **the per-field rules; start here**
5. [Categories: Genres, Themes, Demographics, Tags](#categories-genres-themes-demographics-tags)
6. [Links: sources, streaming_sources, external_sources](#links-sources-streaming_sources-external_sources)
7. [Episode Count](#episode-count)
8. [Synonyms and Deduplication](#synonyms-and-deduplication)
9. [Company Fields — still open](#company-fields--still-open)
10. [Evidence and Its Limits](#evidence-and-its-limits)

---

## What Stage 1 Is

Stage 1 merges the seven provider records for one anime into a single canonical `Anime`, minus the parts other stages own: episodes (stage 2), statistics rollup (stage 4), characters (stage 5). Relationships (stage 3) are merged in the same pass — see below.

**The mappers already canonicalise.** Every `*_anime.jsonl` in an agent directory is already `Anime`-shaped: `status: "ONGOING"`, `type: "TV"`, `rating: "PG-13 - Teens 13 or older"`. Stage 1 is therefore not a parsing problem but a *reconciliation* problem — seven copies of one model that must collapse into one.

The single most important consequence, which nearly every rule below descends from:

> **Which field a value arrives in is a provider's habit, not a fact about the value.**

| Value | One provider files it as | Another files it as |
| :---- | :----------------------- | :------------------ |
| `Shounen` | `demographics` (anilist, mal) | `genres` (animeschedule), `themes` (kitsu) |
| `Super Power` | `genres` (kitsu) | `themes` (anilist) |
| Crunchyroll link | `external_sources` (anidb) | `streaming_sources` (everyone else) |
| `ワンピース` | `title_japanese` (anisearch) | `synonyms` (anilist, kitsu) |

So the merge routes by **what the value is**, never by the field it came in on.

---

## Where the Code Lives

| Module | Holds |
| :----- | :---- |
| `pipeline/metadata_merger.py` | Orchestration and the generic mechanisms. Two entry points: `merge_agent_metadata(agent_dir, offline_data)` reads an agent directory; `merge_provider_records(records, offline_data)` takes records already in memory from `ApiFetcher` |
| `pipeline/metadata_rules.py` | Per-field rules for values and text — episode count, synopsis, titles, categories, synonyms, object fields, month, statistics |
| `pipeline/link_rules.py` | Per-field rules for links, images and media, plus `merged_anime_id` |
| `pipeline/word_lists.py` | MAL's published genre / theme / demographic vocabularies, and `which_field()` |
| `pipeline/relationship_merger.py` | Relations. `merge_provider_records` from here is called by the metadata merger, so one call yields a complete record |

Provider keys are the same service names `ApiFetcher._REGISTRY` uses, so no translation is needed at either entry point.

---

## How the Merge Works

Three mechanisms cover most of the model; the rest are field-specific rules.

**First signal by priority.** For fields every provider agrees on. The default tiebreak is `PROVIDER_PRIORITY` from `relationship_merger.py`:

```
mal > anilist > anidb > anime_planet > anisearch > animeschedule > kitsu
```

Treat that as a tiebreak of last resort, not a ranking of quality. Almost every field that matters wants a *different* winner, which is why the hierarchy table below exists.

Two guards apply:

- **Sentinels never win.** `UNKNOWN` and `OTHER` mean "no data", so a concrete value from the lowest-ranked provider beats `UNKNOWN` from the highest. A sentinel survives only when every provider agrees on it.
- **Empty containers are not values.** Mappers emit a key for every model field, so a field a provider knows nothing about arrives as `{}` or `[]`, not as a missing key. Six providers send `titles: {}`; only AniDB fills it. Without this guard MAL's empty dict outranks AniDB's 27 titles — the one field only AniDB supplies. `provider_supplied()` is the guard; do not replace it with a bare truthiness check on the enum-aware `is_signal`.

Disagreement on a field expected to be uniform is **logged**, because for those fields divergence means a mapper bug rather than a difference of opinion.

**Union.** List fields, deduplicated on a per-field key.

**Sub-field merge.** Object fields are merged one key at a time, never whole-object — see `broadcast` below.

`id` is assigned here rather than by a provider. It is a UUIDv5 derived from the work's own identity, so re-running enrichment over the same anime produces the same id and the record updates in place instead of duplicating. The seed is one provider's canonical key taken in a fixed order, not the whole URL set: the set changes whenever a fetch fails or a provider adds a link, and an id that moves would re-insert the work.

---

## Field Hierarchy

| Field | Rule |
| :---- | :--- |
| `episode_count` | **mal > anidb > anime_planet**, its own order. `0` is read as "not tracked" — it is also the model default, so a provider that does not track episodes is otherwise indistinguishable from one reporting none. See [Episode Count](#episode-count) |
| `synopsis` | **Longest wins** after markup and attribution are stripped, not priority. Priority breaks ties within 10% length. The stripped text is what gets stored |
| `title`, `title_english` | Prefer the non-all-caps variant when the normalised forms match — `One Piece` over `ONE PIECE`. Titles differing by more than case fall back to priority |
| `title_japanese` | Prefer a value in kana or CJK over romaji. Four providers file romaji here and three file `ワンピース`; priority alone fills a Japanese-title field with romaji. The romaji belongs in `synonyms` |
| `month` | AnimeSchedule states it outright; otherwise derived from the merged `aired_dates.aired_from`. Resolved *after* `aired_dates` for that reason |
| `rating` | Sentinel-suppressed, then priority. Only MAL and Kitsu supply a value and they agree (`PG-13 - Teens 13 or older`); a divergence is logged |
| `synonyms` | Union, then the deterministic fold (NFKC, apostrophe variants, punctuation), then drop anything already stated by `title` / `title_english` / `title_japanese`. Runs **last**, because it needs the resolved titles. No embedding model — see [Synonyms](#synonyms-and-deduplication) |
| `genres`, `themes`, `demographics`, `tags` | Classified against MAL's published vocabularies, not trusted from the field they arrived in. Precedence `demographics > genres > themes > tags`. See [Categories](#categories-genres-themes-demographics-tags) |
| `sources` | One URL per work, not one per spelling. Providers and the offline seed are both unioned — neither is complete alone. See [Links](#links-sources-streaming_sources-external_sources) |
| `streaming_sources` | Union keyed on canonical platform + normalised URL. Platform derived from the host where it names the service, else the provider's own label folded |
| `external_sources` | `list[ExternalLink]`. The **residual**: what remains once `sources` and `streaming_sources` have claimed theirs, decided by platform as well as by URL |
| `images` | Each category (`covers` / `posters` / `banners`) unioned separately, deduped only on identical URL. No priority ordering — every provider contributes different artwork, and MAL alone supplies 20 covers. A provider with covers but no banners must not suppress another's banners |
| `trailers` | Union by normalised URL. MAL and Kitsu supply *different* videos for the same anime, so both survive; on a collision the entry with more fields wins, since MAL carries `title` and `thumbnail` where Kitsu carries neither |
| `aired_dates`, `broadcast` | **Sub-field merge, never whole-object.** Providers partition these: mal/anisearch carry `day`/`time`/`timezone`, anilist/kitsu carry `next_episode_at`, animeschedule carries `jp_time`/`sub_time`/`dub_time`/`premiere_*`. Taking the highest-ranked object whole keeps MAL's three keys and discards the other six |
| `statistics` | Provider-keyed dict, plain merge — nothing is arbitrated, since no metric is comparable across platforms. Scale conversion is **not** done here: only a mapper knows its own provider's scale, so `normalize_score` is applied there |
| `nsfw` | **Any provider's `True` wins**, not priority. Only AniList and Kitsu supply it and AniList outranks Kitsu, so priority alone marks a work Kitsu flags as adult "safe". The two errors are not equally bad. Never left empty — `False` when neither supplies it, so the flag is always usable |
| `type`, `status`, `season`, `year`, `duration`, `source_material`, `entity_type`, `country_of_origin` | Uniform across every provider that supplies them. First signal by priority; disagreement logged |
| `background`, `hiatus`, `titles` | Single-source (mal, animeschedule, anidb). First-signal over one candidate is a passthrough; kept in their own list so a second provider appearing later is noticed rather than silently absorbed |
| `opening_themes`, `ending_themes` | MAL only, 30 and 27 entries. Union |
| `content_warnings` | Union. AniList populates it from tags flagged adult, so it is empty for most titles rather than unused |
| `related_anime`, `related_source_material` | Merged by `relationship_merger.py`, called from the same pass |
| `id` | Assigned at merge time, deterministic from the work's identity |
| `score`, `staff_data`, `similarity_score` | Computed or query-time. No provider supplies them |
| `studios`, `producers`, `licensors` | **Open** — see [Company Fields](#company-fields--still-open) |

Coverage note: **AniDB and Kitsu supply no company data at all**, and AniDB supplies neither `genres` nor `duration`. AniDB is the only source of `titles`, and carries by far the most `tags` (122 against AniList's 32).

---

## Categories: Genres, Themes, Demographics, Tags

These four overlap because providers disagree about where a word goes, not about the word. Sorting by the field it arrived in keeps three copies of `Shounen`; sorting by the word keeps one.

### The word lists

From MyAnimeList, read via `https://api.jikan.moe/v4/genres/anime`. That one endpoint returns all 78 terms; split the way MAL splits them:

| Field | Source | Size |
| :---- | :----- | ---: |
| `demographics` | Josei, Kids, Seinen, Shoujo, Shounen | 5 |
| `genres` | MAL's genres plus the three explicit ones | 21 |
| `themes` | MAL's themes | 52 |
| `tags` | everything else | open-ended |

AniList's `Theme-*` tags are deliberately **not** restated in `word_lists.py`. AniList's own mapper already routes them into `themes` by category (`anilist_mapper.py:165-174`), so they arrive sorted and stay themes unless a higher list claims them.

### The four steps

1. **Loosen case and separators**, so `Shounen`, `shounen` and `SHOUNEN` are one word.
2. **Look the word up** in the lists.
3. **Take the first list that holds it**, in the order `demographics > genres > themes > tags`.
4. **Store the list's spelling**, not the provider's.

Step 4 matters as much as the rest. Without it the merge produces `Shounen` *and* `shounen` in `demographics`, and `Action` *and* `action` in `genres` — correctly sorted and still duplicated.

### Why that precedence

| Rank | Field | Why |
| :--- | :---- | :-- |
| 1 | `demographics` | Smallest, most specific list. A demographic must never be left resting as a genre |
| 2 | `genres` | Highest confidence: MAL and AniList agreed on every genre in the sample |
| 3 | `themes` | Larger and fuzzier, so it yields to the two above |
| 4 | `tags` | The catch-all |

Worked examples, with provider placement deliberately disregarded:

| Value | Providers filed it as | On which list | Resolves to |
| :---- | :-------------------- | :------------ | :---------- |
| Shounen | demographic (AniList, MAL), genre (AnimeSchedule), theme (Kitsu) | MAL demographic | `demographics` |
| Super Power | genre (Kitsu), theme (AniList) | MAL theme | `themes` |
| Action | genre ×5, theme (Kitsu) | MAL genre | `genres` |
| Friendship | genre + theme (Kitsu) | none | `tags` |
| Fighting-Shounen | genre (AniSearch) | none | `tags` |

### Shape changes with field

`themes` holds `ThemeEntry` objects; the other three hold plain strings. A word that changes field has to change shape with it — a bare genre promoted to a theme gains a `name` and no description; a theme demoted to a tag loses its description. Miss this and the model rejects the record.

### What it fixes for free

Kitsu's flat `categories` list has no genre/theme distinction, so its mapper emits the same value into both. On One Piece, `action`, `adventure`, `comedy`, `fantasy`, `friendship` and `super power` all arrive as both `kitsu.genres` and `kitsu.themes`. Classifying rather than trusting removes that without special-casing Kitsu.

### Measurement

60 anime across three eras (AniList and Kitsu APIs), plus MAL's published vocabulary. **All 17 AniList genres in the sample were in MAL's genre vocabulary — zero unknown**, and both use the identical five demographics. The disagreement is almost entirely theme-versus-tag: AniList's 154 distinct tags split `Theme 83 / Cast 42 / Setting 15 / Technical 8 / Demographic 5`, and MAL knows 11 of the 83 `Theme-*` tags but 14 of the 70 `Cast`/`Setting`/`Technical` ones — MAL files `Adult Cast`, `School`, `Space` and `Military` as themes where AniList calls them Cast or Setting.

Across the taxonomy check, 272 raw entries folded to 195 distinct, all 195 placed, **0 lost**.

**Limits.** 60 anime, AniList and Kitsu only. MAL's side rests on its published vocabulary rather than crawled data — how often MAL actually populates its Themes row is unmeasured. AniSearch and AnimeSchedule were not sampled beyond One Piece; both supply flat genre lists with no category information, so they are always classified rather than trusted.

---

## Links: sources, streaming_sources, external_sources

Three fields can hold a link to the same place, so ownership has to be explicit.

### Comparison folds the URL, not just the string

`normalize_link_url` produces a comparison key, not a URL to visit. It drops the scheme, `www.`, trailing slashes, percent-encoding and tracking parameters, and folds `x.com` → `twitter.com`. The scheme matters in practice: providers publish the same official site as both `http://` and `https://`, and keeping it filed the Toei page twice.

What it deliberately does **not** collapse:

- **Different paths on one host** — `shingeki.tv` and `shingeki.tv/season1` are different pages; `twitter.com/kimetsu_off` and `twitter.com/DemonSlayerUSA` are different accounts.
- **YouTube `/@handle` vs `/channel/<id>`** — the same channel for One Piece (`@onepieceofficial` = `channel/UCdAHaWcKdpbT5XkN2Er6BUQ`), confirmed by hand. Proving it in general needs a network lookup, and it occurs once in 487 anime; folding it moves link loss from 24% to 23%, so the dependency is not worth taking. Note One Piece still has **two** genuine YouTube channels once that pair is folded — `@onepieceofficial` (Japanese) and `@OnePieceOfficialENG` (English).

### `sources` — one URL per work

Providers decorate the same id differently. MAL publishes `/anime/21/One_Piece` where AniList cross-links `/anime/21`; AnimeSchedule links `anime-planet.com` where Anime-Planet itself says `www.anime-planet.com`. A plain string union keeps all of them: on One Piece that is 13 URLs for 7 works. Identity comes from `canonical_url_key`, and the longest spelling of each work wins — these are work pages with no query strings, so length tracks how much identity the URL states.

**Both halves are unioned because neither is complete.** The offline seed carries livechart, simkl and animecountdown, which no provider returns; the providers carry animeschedule, which the seed does not list.

Kitsu is the one provider addressable two ways — providers report the slug (`kitsu.io/anime/one-piece`), the seed the numeric id (`kitsu.app/anime/12`) — and neither string reveals the other. The numeric form drops when a slug names the same work. MAL and AniList identify every work numerically, so their keys never meet a slug rival and are never dropped. Both slugged and slugless URLs are safe to feed back to the crawlers: MAL follows its own canonical link, and Kitsu resolves a slug via `filter[slug]`.

### `external_sources` is the residual

A link is excluded when another field already owns it, decided **by platform as well as by URL**. URL comparison alone is not enough: AniDB links Crunchyroll as `/series/GRMG8ZQZR` where MAL links `/series-257631`, so the URLs differ and a streaming link would sit in the residual field. The platform does not differ, and that settles it.

On a collision the richer entry wins — MAL states a `label`, AniDB a `language`, and neither should erase the other.

### Why `list[ExternalLink]` rather than a mapping

Measured across 320 anime (AniList 300 popular + 145 random, MAL 22, AniSearch 20), after the full cleanup ladder had already run:

| Source | anime | colliding slots | links lost | per anime |
| :----- | ----: | --------------: | ---------: | --------: |
| AniList, random titles | 145 | 5.7% | 8 | 0.09 |
| AniList, popular titles | 300 | 14.0% | 121 | 0.40 |
| MAL, popular | 22 | 3.3% | 3 | 0.14 |
| AniSearch, popular | 20 | 17.9% | 5 | 0.25 |
| **All three merged** | **20** | **20%** | **29 (24% of all links)** | **1.45** |

**Popularity drives it** — a sample of random titles understates the problem roughly four-fold. **Merging multiplies it** — each provider alone looks tolerable (3–18%), but merged the rate more than doubles, because providers hold *different* accounts for the same platform. MAL has `@heroaca_anime`, AniList has `@MHAOfficial`, and both are real.

Almost all collisions are language or region variants of one thing:

```
Chainsaw Man, twitter   CHAINSAWMAN_PR  Chainsaw_EN  Chainsaw_FR
                        ChainsawMan_PT  chainsawman_es  chainsawman_la
Demon Slayer, twitter   kimetsu_off (JP)  DemonSlayerUSA  kimetsu_fr
```

AniList tags each link with a language, which separates 81% of collisions cleanly — but not enough on its own: 4% are two links sharing one language (Chainsaw Man has both `_es` and `_la` Spanish accounts), 15% have no language, and only AniList supplies it at all.

### Platform comes from the host, never the label

Providers key by completely different things, which is why the label cannot be the platform name:

| Provider | What the key actually is | Example |
| :------- | :----------------------- | :------ |
| AniList | generic category | `Official Site`, `Twitter` |
| MAL | the account handle | `@MHAOfficial`, `@heroaca_anime` |
| AniSearch | the page's own title | `Toei Animation`, `Kimetsu`, `Gkids` |
| AniDB | the platform | `twitter`, `allcinema` (already canonical) |

AniSearch's and MAL's keys are per-link titles, not platform names. Those titles carry real information — `Toei Animation` vs `One Piece` distinguishes the studio's page from the show's — which is what `ExternalLink.label` is for.

The same reasoning applies to `streaming_sources`, where the host names the service. The exception is AnimeSchedule's affiliate shorteners (`amzn.to`, `apple.co`), whose host names nothing, so the provider's own label is folded instead.

### Routing must not depend on which provider spoke

The same URL can be classified differently by different providers. AniList tags both One Piece YouTube channels `type: STREAMING` while MAL supplies `@onepieceofficial` with no type at all. Measured: of 57 links supplied by more than one provider, **1 (2%)** routed inconsistently — rare, but it makes output depend on provider order, which is the same defect as a first-writer-wins merge.

---

## Episode Count

**`mal > anidb > anime_planet`, and the field holds whatever the winning provider reports.** The merge does not reconcile what the number counts — which matters, because it does not mean the same thing everywhere.

### MAL does not simply count aired episodes

That is true for open-ended shows and false for everything else:

| Show | MAL anime page | Episode-list counter | Stored | Which quantity |
| :--- | :------------- | :------------------- | -----: | :------------- |
| One Piece (open-ended) | `Unknown` → 0 | `(1174/Unknown)` | 1174 | **aired** |
| Tsuihou Sareta (airing cour) | 26 | `(12/26)` | 26 | **announced total** |
| Seitokai ni mo Ana wa Aru! (not started) | 12 | `(0/12)` | 12 | **announced total**, nothing aired |
| Aoashi 2nd Season (not started) | 0 | `(0/Unknown)` | 0 | nothing known |

The episode-list fallback fires only when the anime page is empty, so the aired-only path is reached solely for shows with no announced total. For a cour show the stored number is forward-looking — including a count of 12 for a show where nothing has aired.

If an aired-episode count is ever wanted it does not need a provider: stage 2 emits episode records carrying `aired`, so counting those with a past date gives it directly.

### What each provider reports, for One Piece

| Source | Count | Note |
| :----- | ----: | :--- |
| MAL episode-list counter | 1174 | reached only because One Piece has no announced total |
| AniDB `<episodecount>` | 1184 | includes scheduled and undated episodes |
| Anime-Planet | 1179 | |
| AniList, AniSearch, AnimeSchedule, Kitsu | 0 | do not track it for a long-running series |
| offline seed | 1155 | frozen — upstream archived |

The seed is not used for this field: its upstream is archived, so it can only grow staler.

### Seed database status

`manami-project/anime-offline-database` and its generator `modb-app` were archived in early July 2026. The final release is `2026-27` (41,537 entries, 2026-07-04) and remains downloadable, since archived repositories keep release assets.

An active continuation exists: [`cedya77/anime-offline-database`](https://github.com/cedya77/anime-offline-database) — weekly releases, identical schema, and more provider IDs than manami's final on every source (it merges duplicate entries, so the entry count is lower while ID coverage is higher). [Fribb/anime-lists](https://github.com/Fribb/anime-lists) already sources from it. The seed can be refreshed rather than abandoned; it remains the weakest source either way.

---

## Synonyms and Deduplication

Two kinds of duplicate survive a plain union, and both are handled deterministically.

**Cosmetic variants.** The same title with different punctuation or width — `All'arrembaggio!` arrives with three different apostrophes (U+0027, U+0060, U+2019). NFKC plus apostrophe folding collapses them. Measured on the real synonym pool: 92 raw → 34 under the old `.lower().strip()` → **32** with NFKC and apostrophe folding.

**Title bleed.** The work's own title appears among its synonyms — AniDB lists `One Piece` as a synonym of One Piece. This is why synonym dedup runs **late**: it filters against the resolved `title` / `title_english` / `title_japanese`, so it cannot run before they are decided.

### No embedding model, and this is not a cost decision

Measured over 393 synonym values from 40 anime, comparing the deterministic ladder against `deduplicate_semantic_array_field` with BGE-M3. Every value the model removed that plain normalisation kept was classified by hand against its closest surviving match; a removal is wrong when the two denote different things.

| Approach | Removed | Wrong | Breakdown |
| :------- | ------: | ----: | :-------- |
| **deterministic ladder** | **88** | **0** | — |
| semantic @ 0.80 | 160 | 32 | 17 season, 10 script, 5 format |
| semantic @ 0.85 (shipped default) | 125 | 14 | 5 season, 5 script, 4 format |
| semantic @ 0.90 | 90 | 6 | 4 format, 1 script, 1 season |
| semantic @ 0.95 | 51 | 1 | 1 season |
| semantic @ 0.98 | 26 | 0 | — |

The model loses on both axes at once. At equal recall (0.90, ~90 removals) it makes 6 wrong merges against 0. At equal precision (0.98, no wrong merges) it removes 26 against 88 — 3.4x fewer. **No threshold is ahead on both.**

Real wrong merges:

```
Mask Danshi…        dropped 'マスク男子は恋したくないのに OAD'   the OAD is a separate release
Minky Momo          dropped 'مغامرات حنين 2'                   season 2, while season 1 was kept
The New Gate        dropped 'ザ・ニュー・ゲート'                 the entire Japanese title
Renzu               dropped 'Рензу: Растояние между двумя'     the only Russian title
Jing Cui Xian Zun   dropped 'Jing Cui Xian Zun: Part II'       Part 2 kept, Part II dropped
```

It also dropped `The Legend of Heroes: Trails in The Sky` — the official English title — in favour of the romaji. For a search index that is the most valuable synonym in the list.

**Why it fails is structural, not a tuning problem.** Embeddings encode meaning, and "X Season 1" and "X Season 2" mean nearly the same thing: the distinguishing token is a single digit inside a 1024-dimensional vector. Anime synonyms are precisely the case where small differences carry the information — season and part numbers, `OAD`/`OVA`/`Movie` markers, script variants. String folding never touches those.

For completeness: the ladder runs at 0.03 ms per anime against 113 ms for the semantic pass, plus a 6.4 GB model — roughly 1.3 seconds against 75 minutes across 40,346 anime. But the accuracy result alone settles it.

**Two things worth knowing before anyone revisits this.** `_is_semantically_duplicate` binds its threshold as a *default argument*, fixed at import time, so reassigning `SEMANTIC_SIMILARITY_THRESHOLD` at runtime has no effect. And `deduplicate_synonyms_language_aware` falls back to plain dedup when no model is injected, silently — it is a no-op in this pipeline today.

**A genuine win left on the table.** Most of what the model legitimately caught is transliteration noise: `Tēkyū` vs `Teekyuu`, `Mahō` vs `Mahou`. Japanese long vowels are written either with a macron or by doubling, so folding both is deterministic — strip diacritics, then collapse `ou`/`oo` → `o` and doubled vowels to single. That takes the ladder from 72 removals to 88 and cannot merge two different works. It belongs in the fold, not in a new tier.

**Limits.** 40 anime, 393 values, one seed. The wrong-merge classification is a hand judgement, not independent ground truth. The comparison used `deduplicate_semantic_array_field`, which does no language grouping; the language-aware variant would remove the SCRIPT class — 1 of the 6 errors at 0.90, leaving 5 against 0.

### If language-aware dedup is ever wired

Use `titles`, not `langdetect`. AniDB ships `titles` as an ISO-keyed map (`de`, `ru`, `ko`, `th`, `tr`, `zh-Hant`, …), which is ground truth. Asking `langdetect` for the language of the two-character string `海贼` is unreliable by construction — short strings are its documented weak case, and a misgroup silently drops a real variant. There is a genuine win case: within a Greek group, `Ντρέηκ, το Κυνήγι του Θησαυρού` and `Ντρέικ και το Κυνήγι του Θησαυρού` differ by one letter plus a conjunction. Correctly *not* collapsed: `海贼` ("sea thief") and `海贼王` ("sea thief king") are distinct titles despite high lexical overlap — which is exactly why the within-language constraint matters.

---

## Company Fields — still open

`studios`, `producers` and `licensors` are not merged. Two problems have to be settled together.

**Role is a property of the work, not the company.** In the offline database's 40,346 entries, 1,219 names (18%) appear as both studio and producer — `toei animation` is 391 studio / 1047 producer, `production i.g` 255 / 454. On One Piece, 4 of 5 companies land in different fields depending on the provider. So a company cannot be filed by role once and for all.

Related: our AniList mapper splits on `isAnimationStudio`, a *company* property, to populate per-work role fields, while `isMain` — the per-work signal — is fetched and discarded (`anilist_mapper.py:176-187`, `anilist_helper.py:312-320`).

**The same company arrives under different names.** AniSearch writes `Toei Animation Co., Ltd.` where the others write `Toei Animation`. At scale, 703 names in the offline database merge under legal-suffix folding, affecting 38,381 credits — though that dataset is already merged and lowercased, so it is not a fair proxy for what stage 1 receives.

The fold itself is straightforward: NFKC, case, punctuation, plus legal suffixes only (`Co., Ltd.`, `Inc.`, `K.K.`, `Corp.`, `LLC`). Do **not** strip industry words (`studio`, `production`, `entertainment`) — that merged `tsuburaya entertainment` with `tsuburaya productions`, which are distinct entities.

**What is undecided** is the shape: union per field, one provider decides, or a single `companies` list carrying roles. The last is a model **and proto** change.

**Coverage.** AniDB and Kitsu supply no company data at all.

**Limits.** The role and name-variation counts come from the offline database; the raw cross-provider comparison is still One Piece alone. An earlier claim that provider company names match exactly was drawn from three providers on one title and is wrong.

---

## Evidence and Its Limits

Field-behaviour figures are measured against the agent directories, of which `One_agent2` through `One_agent5` each hold all seven providers. All four produce identical output, so that is **one work (One Piece) captured four times, not four samples** — enough to design against, not enough to generalise. Re-check once a second title has a complete run.

Three areas are measured more widely:

| Area | Sample |
| :--- | :----- |
| [External sources](#links-sources-streaming_sources-external_sources) | 320 anime — AniList 445, MAL 22, AniSearch 20, AniDB 133 |
| [Synonym dedup](#synonyms-and-deduplication) | 393 values from 40 anime in the offline database |
| [Categories](#categories-genres-themes-demographics-tags) | 60 anime across three eras, AniList and Kitsu |

### Defects this investigation surfaced

Recorded because each was silent, and each is the kind of thing that can return:

- **MAL was skipped for 29,863 of 29,864 anime.** `fetch_all` rejected bare `/anime/{id}` URLs, and the seed contains almost nothing else. The crawler now reads the page's own canonical link.
- **MAL's episode count never reached disk.** The record was persisted before the fallback patched `episode_count`, so callers saw 1174 while the file kept 0.
- **AnimeSchedule's `month` was fetched, parsed and dropped.** The API returns `"month": "October"` and the source model declared the field; the mapper never mapped it.
- **AniSearch's score was merged on the wrong scale.** It rates out of five stars (the page states `Calculated Value 4.18 = 84%`) while every other provider reports out of ten, so its score merged at roughly half. Converted in the mapper, where the provider's scale is known.
- **`.env` is not loaded by anything in the codebase.** AniDB falls back to `os.getenv("ANIDB_CLIENT", "animeenrichment")`, a name AniDB rejects with `error code="302"`. Any run not launched from a pre-sourced shell fetches AniDB with an invalid client and gets nothing.
- **The pipeline reported `Success Rate: 100.0%` in the same run whose summary showed `anidb: ✗`.**
- **`normalize_mal_anime_url` still returns `has_slug`**, which no caller uses — dead since the crawler started resolving the canonical URL itself.

### Superseded claims

Kept so nobody re-derives them from stale notes:

- An earlier figure said a mapping "must discard roughly 6 real links" for One Piece. That was measured before URL normalisation and routing; most were the same channel written two ways, or trailers. The real residual is two.
- An earlier note predicted a regenerated fixture would show **fewer** AniDB keys after the resource-type fix. The opposite happened — it raised AniDB's contribution by 33% (492 → 652 entries across 101 anime).
- `rating` previously disagreed across providers (`PG-13` vs `PG - Children` from Kitsu). On data refetched 2026-09-21 they agree; the merge logs a warning if they diverge again.
