# Merge Rules

**Last updated**: 2026-09-24
**Purpose**: How seven provider records become one canonical `Anime`, field by
field, and the evidence behind every rule that is not obvious.

Seven providers describe one work. Each field needs a rule for which value
survives, and most of those rules look arbitrary until you know what was measured
to choose them. This document is that record, so nobody re-measures what has been
measured and so a constant that looks wrong is not "tidied up" back into a bug.

What each provider publishes, and where, is in
[source_api_field_mappings.md](source_api_field_mappings.md). This document is
only about reconciling them. It describes what is true now; it is not a history
of how the rules got here.

---

## Table of Contents

1. [What The Merge Does](#what-the-merge-does)
2. [Where The Code Lives](#where-the-code-lives)
3. [How The Merge Works](#how-the-merge-works)
4. [Field Hierarchy](#field-hierarchy) — the contract, start here
5. [Categories](#categories) — genres, themes, demographics, tags, content warnings
6. [Companies](#companies) — studios, producers, licensors
7. [Score](#score)
8. [Links](#links) — sources, streaming, external
9. [Synonyms](#synonyms)
10. [Episode Count](#episode-count)
11. [Relationships](#relationships)
12. [Evidence and Its Limits](#evidence-and-its-limits)

---
## What The Merge Does

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
## Where The Code Lives

| Module | Holds |
| :----- | :---- |
| `pipeline/metadata_merger.py` | Orchestration and the generic mechanisms. Two entry points: `merge_agent_metadata(agent_dir, offline_data)` reads an agent directory; `merge_provider_records(records, offline_data)` takes records already in memory from `ApiFetcher` |
| `pipeline/metadata_rules.py` | Per-field rules for values and text — episode count, synopsis, titles, categories, synonyms, object fields, month, statistics, score |
| `pipeline/link_rules.py` | Per-field rules for links, images and media, plus `merged_anime_id` |
| `pipeline/same_word.py` | When two differently-written category words are the same word |
| `pipeline/same_company.py` | When two differently-written company names are the same company |
| `pipeline/relationship_merger.py` | Relations, and `PROVIDER_PRIORITY`. Its `merge_provider_records` is called by the metadata merger, so one call yields a complete record |

Provider keys are the same service names `ApiFetcher._REGISTRY` uses, so no
translation is needed at either entry point.

---
## How The Merge Works

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
| `synonyms` | Union, then the deterministic fold (NFKC, apostrophe variants, punctuation), then drop anything already stated by `title` / `title_english` / `title_japanese`. Runs **last**, because it needs the resolved titles. No embedding model — see [Synonyms](#synonyms) |
| `genres`, `themes`, `demographics`, `tags`, `content_warnings` | One word in exactly one field, decided by the most trusted provider that classified it. See [Categories](#categories) |
| `sources` | One URL per work, not one per spelling. Providers and the offline seed are both unioned — neither is complete alone. See [Links](#links) |
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
| `related_anime`, `related_source_material` | See [Relationships](#relationships) |
| `id` | Assigned at merge time, deterministic from the work's identity |
| `score` | Computed from provider scores and vote counts — mean, median and a confidence-adjusted figure. See [Score](#score) |
| `staff_data` | Computed. No provider supplies it |
| `studios`, `producers`, `licensors` | One `companies` list carrying roles, since a company can hold two on one anime. See [Companies](#companies) |

Coverage note: AniDB supplies neither `genres` nor `duration`. AniDB is the only source of `titles`, and carries by far the most `tags` (122 against AniList's 32).

---
## Categories

Genres, themes, demographics, tags and content warnings overlap because providers
disagree about **where** a word goes, not about the word. `Shounen` arrives as a
demographic from MAL and AniList, a genre from AnimeSchedule, a theme from Kitsu
and a tag from AniDB. Keeping all four stores one word four times.

### The rule

**The most trusted provider that actually classified the word decides.**

A `tags` entry is not a classification — it means the provider had nothing to
say. AniDB has no genre, theme or demographic field at all, so everything it
knows arrives as a tag. Letting that count would demote `Swordplay` out of themes
on AniDB's say-so alone.

Two qualifications:

- **`content_warnings` overrides everything.** Any provider flagging a word as
  adult content wins outright, whoever disagrees. AniList populates it from tags
  flagged adult, so it is empty for most titles rather than unused.
- **`genres > demographics > themes > tags`** settles a single provider that used
  two of its own fields — Kitsu files `Super Power` under both its genres and its
  themes.

A word nobody classified stays a tag.

**There is no vocabulary of our own.** The providers decide. Maintaining a list
of which words are genres would mean deciding, for every new word any provider
invents, where it belongs — and being wrong silently whenever a provider knows
better than the list.

### Which spelling survives

The one most providers used, with priority breaking ties. AniDB writes everything
lowercase and would otherwise decide the storage form for every word it touches.

Spellings differing beyond case are handled by `same_word.py`, which layers a
hand-checked table over the shared fold: `science fiction` / `sci fi`,
`superpowers` / `super power`, `cross dressing` / `crossdressing`, `iyashi kei` /
`iyashikei`.

**Each of the twelve entries was verified by hand, and that is the point.**
Close-looking words are usually not the same word: `France` scores 77% against
`Romance`, `sentai` 83% against `Hentai`, and `Shounen Ai` 82% against `Shounen`
while meaning something else entirely. Of 26 candidates a similarity threshold
proposed, **24 were wrong**. A threshold cannot be tuned into correctness here.

### Shape changes with the field

`themes` holds `ThemeEntry` objects; the other three hold plain strings. A word
that changes field has to change shape with it — a bare genre promoted to a theme
gains a `name` and no description; a theme demoted to a tag loses its
description. Miss this and the model rejects the record.

### What it fixes for free

Kitsu's flat `categories` list has no genre/theme distinction, so its mapper
emits the same value into both. On One Piece, `action`, `adventure`, `comedy`,
`fantasy`, `friendship` and `super power` all arrive as both `kitsu.genres` and
`kitsu.themes`. Deciding by who classified rather than trusting the field removes
that without special-casing Kitsu.

### Measured

30 anime across all seven providers: 2,450 values, 1,556 distinct after folding,
**0 lost, 0 placed in two fields, 0 order-dependent**. Content warnings: 154 of
154 landed in `content_warnings`, with no other field changed.

**Limits.** The order-independence check matters because an earlier design was
sensitive to which provider was read first; it is verified through
`merge_provider_records` rather than against the rule in isolation.

---
## Companies

One list, each entry carrying every role any provider gave it and every link:

```
companies: [
  {name: "Toei Animation", roles: [STUDIO, PRODUCER], sources: [7 urls]},
  {name: "Fuji TV",        roles: [PRODUCER],         sources: [3 urls]},
  {name: "Funimation",     roles: [LICENSOR],         sources: [2 urls]},
]
```

Three separate lists cannot express this. A company holding two roles on one
anime is not an edge case: Kitsu files Madhouse as both producer and studio on
Death Note, AniDB files Toei Animation as both on One Piece, and **23% of
company-anime pairs have providers disagreeing about the role**.

`enrichment/pipeline/same_company.py`, three layers:

| Layer | Catches |
| :--- | :--- |
| `fold_for_comparison` — case, NFKC, punctuation. Shared with categories and synonyms | `MADHOUSE` / `Madhouse`, `P.A.WORKS` / `P.A. Works` |
| legal suffixes — `Co., Ltd.`, `Inc.`, `K.K.`, `Corp.`, `LLC`, `GmbH`, `Company` | `Toei Animation Co., Ltd.` / `Toei Animation`, `Gonzo K.K.` / `GONZO` |
| plurals, paired | `Ashi Production` / `Ashi Productions` |

**Industry words are deliberately not stripped.** Removing `production` or
`entertainment` merges `Tsuburaya Entertainment` with `Tsuburaya Productions`,
which are different companies.

**Plurals only fire when both spellings are present** among the names being
compared. Dropping a trailing `s` unconditionally turns `BONES` into `bone`,
which is safe today only because no company is called `Bone`. Over Kitsu's 1,975
producers the pairwise rule merges exactly the same 54 groups as the blunt rule
while altering no other key.

**Plurals cannot live in `fold_for_comparison`.** That fold is shared: adding
plurals there moved categories on 12 of 75 sample anime and broke the
hand-checked `superpowers` → `super power` entry in `same_word`, to gain two
company merges.

**The spelling the most providers used.** On a tie — including when every
spelling comes from a single provider — the highest-ranked provider in
`PROVIDER_PRIORITY` decides.

Measured over 232 groups with more than one spelling: counting settles 193, and
the hierarchy is needed for 39, of which 33 are one-provider-each. The fallback
is common enough to need specifying, not incidental.

Providers fall into two kinds, and they are not equally informative:

- **Multi-role**: MAL, AniList, Kitsu, AniDB. These have somewhere to put a
  producer, so filing a company as a studio is a decision.
- **Single-category**: Anime-Planet, AniSearch, AnimeSchedule. One studio field
  each. Filing a company as a studio is the only thing they can do.

```
any multi-role provider covers the company  ->  union of what THEY say
no multi-role provider covers it            ->  the single-category label
```

The single-category label is right 82–97% of the time — checked against what the
multi-role providers say about the same company on the same anime — but when it
is wrong it is wrong in a specific way: `Fuji TV` is a broadcaster, `GKIDS` and
`Discotek Media` are distributors, and none of them animate anything.

**Validated against companies whose type is public fact**, not against the
providers themselves. Over 339 appearances of companies that never animate
(distributors, broadcasters, publishers):

| Rule | Wrongly marked studio |
| :--- | ---: |
| multi-role decides, single-category as fallback | **2 (0.6%)** |
| union everything | 7 (2.1%) |

Deferring to the multi-role providers removes five of seven misclassifications.
The two it cannot fix are companies only a single-category provider knows about,
where its label is the only evidence there is.

**Among multi-role providers, disagreement is unioned rather than arbitrated.**
They each had a real choice, so `mal: studio` with `kitsu: producer` is two
providers reporting two genuine roles, not one of them being wrong.

Union across providers, normalised through `link_rules.normalize_link_url`. All
seven supply a company URL, so a well-covered company carries seven links, which
is far stronger evidence of identity than the name.

### One company, many spellings, one role

One Piece, `Toei Animation`. Six providers write it plainly, AniSearch writes
`Toei Animation Co., Ltd.`

```
mal            studio    Toei Animation             myanimelist.net/anime/producer/18/...
anilist        studio    Toei Animation             anilist.co/studio/18
kitsu          studio    Toei Animation             api/edge/producers/8
animeschedule  studio    Toei Animation             animeschedule.net/studios/toei-animation
anime_planet   studio    Toei Animation             anime-planet.com/anime/studios/toei-animation
anisearch      studio    Toei Animation Co., Ltd.   anisearch.com/company/412,...
anidb          studio    Toei Animation             anidb.net/creator/1834
                                                  -> also producer, from AniDB's Work type

{name: "Toei Animation", roles: [STUDIO, PRODUCER], sources: [7 urls]}
```

Rule 1 folds the legal suffix. Rule 2 picks `Toei Animation`, six providers to
one. Rule 3 unions STUDIO and PRODUCER, both from multi-role providers. Rule 4
keeps all seven links, including AniSearch's, which points at the same company
under its longer name.

### One company, two roles, from one provider

Death Note, `Madhouse`. Kitsu lists it twice on the same anime under different
roles, which is a fact about the work rather than a duplicate.

```
kitsu    studio     Madhouse   (producer id 917)
kitsu    producer   Madhouse   (producer id 917)
mal      studio     MADHOUSE
anilist  studio     MADHOUSE

{name: "MADHOUSE", roles: [STUDIO, PRODUCER], sources: [...]}
```

Note Rule 2 picks `MADHOUSE` here, two providers to one — the opposite casing to
the Toei example, because the rule follows the providers rather than a
preference.

### A single-category provider overruled

`GKIDS` is a distributor. AniSearch has only a studio field, so that is where it
puts it; the multi-role providers know better.

```
anisearch  studio     GKIDS      <- single-category, ignored for role
mal        licensor   GKIDS
kitsu      producer   GKIDS

{name: "GKIDS", roles: [LICENSOR, PRODUCER], sources: [...]}
```

Under "union everything" this would be `[LICENSOR, PRODUCER, STUDIO]`, claiming a
distributor animates.

### Nobody but a single-category provider knows it

`Crunchyroll` on *Double Hard*. No multi-role provider lists it, so its label
stands even though it is wrong.

```
animeschedule  studio   Crunchyroll

{name: "Crunchyroll", roles: [STUDIO], sources: [...]}
```

This is the residual 0.6% error. Correcting it would need a company-type list we
have no basis for and would have to maintain.

### A tie on spelling

`5-Okunen Button`. Two spellings, one provider each, so counting decides nothing
and `PROVIDER_PRIORITY` does.

```
mal        STUDIO SOTA
anisearch  Studio Sota

{name: "STUDIO SOTA", ...}      mal outranks anisearch
```

### A plural pair

Death Note, from Kitsu's two separate producer records.

```
Ashi Production    (producer 1455)
Ashi Productions   (producer 266)

{name: "Ashi Productions", ...}
```

Both forms are present, so the plural rule pairs them. `BONES` appearing alone
on another anime is untouched, because nothing there claims to be `Bone`.

- **0.6% of never-animating companies are marked studio**, where a
  single-category provider is the only source. Accepted; the alternative is a
  maintained company-type list.
- **Multi-role disagreement is unioned, not resolved.** `mal: studio` against
  `kitsu: producer` on `Arcturus` may be two real roles or two sites differing.
  We cannot tell them apart and do not try.
- **The plural rule would merge two genuinely different companies** if one were
  the literal plural of the other. None exists in 1,975 Kitsu producers or 684
  anime, but nothing prevents it.
- **AniDB's coverage is understated** in every measurement here. It was banned
  throughout collection and served only cached responses.

---
## Score

### What we publish

Three numbers, each answering a different question.

| Field | What it is | What it answers |
| :--- | :--- | :--- |
| `mean` | Plain average of the provider scores, every provider counting once | What do the sites say? |
| `median` | Middle provider score | What does the typical site say, ignoring an outlier? |
| `weighted` | Confidence-adjusted score (below) | What would we stand behind, given how many people actually voted? |

`mean` and `median` stay on the providers' own scale, so they are comparable
with a MAL or AniList number. `weighted` is deliberately not — it is the number
to **sort by**, not the number to show beside a provider's.

All three are computed from provider scores already normalised to 0–10 by each
mapper. AniSearch rates out of five and is doubled; AniList and Kitsu publish
0–100 and are divided by ten.

### The weighted score

Every anime is handed a batch of imaginary votes, all set to the score a typical
anime gets. Real votes and imaginary votes go in one pot, and we take the
average.

```
weighted = (V * M + m * C) / (V + m)

  M  plain average of the provider scores
  V  total voters, summed across every provider that reports a count
  C  6.2  — what the imaginary voters say ("assume it is ordinary")
  m  1000 — how many imaginary voters there are
```

**Why bother.** An anime scored 9.5 by twelve people is not better than one
scored 8.6 by four hundred thousand. `mean` and `median` cannot tell those
apart — both just see 9.5 and 8.6. The vote count is the only thing that can.

**How it behaves.** When real votes are few, the imaginary ones outnumber them
and the result sits near ordinary. As real votes arrive, the imaginary batch is
outnumbered and the anime's own score decides the result. Nothing is hidden and no
threshold is crossed; the correction simply fades.

Same anime, every provider scoring it 9.00, only the voter count changing:

| real voters | weighted | share of the pot that is real |
| ---: | ---: | ---: |
| 10 | 6.23 | 1% |
| 100 | 6.45 | 9% |
| 1,000 | 7.60 | 50% |
| 5,000 | 8.53 | 83% |
| 20,000 | 8.87 | 95% |
| 200,000 | 8.99 | 100% |

The clearest real case in our sample is *Pinocchio no Bouken*: AniDB 2.0 from
two voters, AniSearch 1.0 from one. The plain mean is **1.50**, which would rank
it among the worst anime ever made on the word of three people. Weighted, it is
**6.19** — we do not know, so it sits at ordinary.

Measured across 93 titles with all seven providers attempted:

| | |
| :--- | ---: |
| median movement from the plain mean | 0.25 |
| 90th percentile | 1.05 |
| largest | 4.70 |
| moved more than 0.25 | 48% of titles |
| moved more than 0.50 | 24% |
| moved more than 1.00 | 11% |

The formula is the standard weighted-rating shape used by IMDb (for its Top 250)
and by MAL, which applies it with a batch of 50 imaginary votes. We are not
inventing a scheme; we are choosing its two constants for our data.

### Why not weight the providers themselves

The obvious reading of "weighted" is to let providers with more voters count for
more. **We do not do this, and the reason is measured.**

Weighting providers in proportion to their voter counts, across seven titles:

| scheme | MAL | AniList | Kitsu | AniSearch | biggest : smallest |
| :--- | ---: | ---: | ---: | ---: | ---: |
| equal | 25.0% | 25.0% | 25.0% | 25.0% | 1x |
| by votes | 78.3% | 15.4% | 6.0% | 0.3% | **286x** |
| by square root of votes | 56.9% | 25.2% | 14.6% | 3.3% | 17x |
| by log of votes | 30.6% | 27.0% | 24.2% | 18.2% | 2x |

Weighting by votes hands MAL 78% of the decision and AniSearch 0.3%. "Our score"
would be MAL's score with a rounding error, and adding an eighth provider would
change nothing, because any new source is small next to MAL. That defeats the
point of merging at all.

Softening it does not rescue the idea. How far each scheme lands from the plain
mean:

| scheme | average distance from the mean | range |
| :--- | ---: | :--- |
| by votes | +0.142 | −0.01 to +0.30 |
| by square root | +0.088 | −0.05 to +0.21 |
| by log | +0.016 | −0.02 to +0.05 |

The tighter the imbalance is controlled, the more the result collapses back into
the plain mean — log weighting is a third of a rounding error away from it, so
we would be storing two copies of one number.

**The conclusion we settled on:** vote counts should not decide *which site* to
believe. They should decide *whether to believe the number at all*. Every
provider counts once toward `M`; the votes only control how far `M` is allowed to
differ from the ordinary score.

This also keeps a property we want: adding a provider genuinely moves the score,
because it changes both `M` and `V`.

### Why C is 6.2

`C` is the score the imaginary voters give. It decides **which score the formula
leaves alone** — an anime already scoring `C` is unchanged no matter how few
votes it has, and everything else is pulled toward it.

So the question is: which anime deserve to be left alone? The ordinary ones. `C`
must therefore be what an ordinary anime actually scores, measured on our own
data.

| source | value |
| :--- | ---: |
| our seven providers, sample reweighted to the database's real shape | **6.22**, used as **6.2** |
| offline database `arithmeticMean`, 29,197 scored entries | 6.21 |

The two agree, but we use our own figure. The offline database is built from a
different set of sites — it includes animecountdown, simkl, animenewsnetwork and
livechart, which we do not use, and excludes animeschedule, which we do — so it
is not our pool and must not be treated as our benchmark.

### Why not 5

5 looks neutral because it is the middle of the scale. It is not neutral, because
the imaginary votes are not a placeholder — they are a guess at what this anime
probably scores, and the best guess is what comparable anime score.

| voters | providers say | C = 6.2 | C = 5.0 |
| ---: | ---: | ---: | ---: |
| 50 | 6.20 | **6.20** | 5.06 |
| 300 | 6.20 | **6.20** | 5.28 |
| 200 | 7.50 | 6.42 | 5.42 |
| 20,000 | 8.00 | 7.91 | 7.86 |

Row one is the argument. Every provider calls the anime dead average, and with
`C = 5` we publish 5.06 — a full point of dissatisfaction nobody expressed.
Across 93 real titles, switching to 5 moves scores by **0.63 on average, always
downward**. That is not withholding judgement, it is a standing penalty on
anything obscure.

### Maintaining it

`C` belongs in config, not in code. See
[Recalculating C and m](#recalculating-c-and-m) for when and how it is refreshed.

### Why m is 1000

`m` is how many imaginary voters there are — the dial for how much evidence an
anime needs before its own score wins. At exactly `m` real voters the pot is half
real and half imaginary, so the result sits midway between the anime's score and
ordinary.

| m | effect |
| ---: | :--- |
| 100 | too trusting; a hundred voters is enough to be believed outright |
| **1000** | chosen; leaves anything above roughly 20,000 voters untouched, corrects the thin end |
| 5000 | too harsh; pulls Cowboy Bebop from 8.60 down to 8.53 on 170,305 voters, which is wrong |

For scale: MAL uses 50 and IMDb uses 25,000. Ours sits between because we pool
votes from all seven providers, so totals are larger than any single site's, but
our database reaches far more obscure titles than IMDb's Top 250 ever considers.

`m = 1000` is a judgement, not a measured optimum. Anywhere from roughly 500 to
2000 is defensible; the value was chosen because 5000 is demonstrably too harsh
and 100 too permissive. It is settled, but it is the least evidence-backed number
in this document.

Like `C`, `m` belongs in config, and should be revisited when the provider set
changes — more providers means larger `V`, so titles clear `m` more easily and the
correction weakens across the board. That is correct behaviour (more evidence,
more confidence), but it is a behaviour change that should be noticed, not
absorbed silently.

### When we publish nothing

The formula returns exactly `C` when `V` is zero. That number is a lie — we did
not measure a brand-new anime and find it average, we measured nothing. Publishing
6.2 would claim a finding and would park every unaired show mid-table.

| situation | mean / median | weighted |
| :--- | :--- | :--- |
| no provider reports a score | absent | absent |
| scores exist, no vote counts anywhere | computed | **absent** |
| scores exist with vote counts | computed | computed |

This is not an edge case. A currently-airing show enters the database with scores
from several providers long before anyone has rated it.

**Related defect, fixed 2026-09-23.** AniSearch prints `Calculated Value 0.00 =
0%` for unrated anime, with all five star rows at zero votes. We were storing
that as a score of nought. Eight records in a 110-anime sample, and on one title
AniSearch was the only provider, so our merged score would have been 0.00 —
bottom of the database for an anime nobody had rated. The crawler now ignores a
zero score.

### Recalculating C and m

**Not implemented.** Nothing enriched at scale yet, so there is no populated
database to compute a new `C` from. This section is the design to pick up when
there is.

Both are settings, not constants. `C` is what a typical anime scores, and that
moves as the database grows and as providers are added.

### When

**Every 5,000 newly enriched anime**, and additionally whenever the provider set
changes. Adding a provider shifts `C` directly, and it raises the pooled vote
total `V`, so titles pass `m` more easily and the correction weakens across the
board.

Do not recompute on every enrichment run. Scores would drift slightly every time
and the database would never settle.

### How

1. Take the `mean` of every anime in the database that has one.
2. Average those. That is the new `C`.
3. Write it to config with the date and the number of anime it came from.
4. Recompute `weighted` for every record. This is a payload rewrite, not a
   re-embedding, so vectors are untouched.

`m` does not need recomputing from data — it is a judgement about how much
evidence a title needs before its own score is trusted. Revisit it only when the
provider set changes enough to move typical vote totals.

### What recalculating does to existing scores

An anime's score moves by `ΔC * m / (V + m)`, where `ΔC` is how far `C` moved.
The movement can never exceed `ΔC` itself, and shrinks as real votes grow:

| voters | C moves 0.2 | C moves 0.3 |
| ---: | ---: | ---: |
| 50 | 0.190 | 0.286 |
| 1,000 | 0.100 | 0.150 |
| 5,000 | 0.033 | 0.050 |
| 20,000 | 0.010 | 0.014 |
| 100,000 | 0.002 | 0.003 |

Measured across 720 real anime:

| change in C | average movement | largest | moved more than 0.05 |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.035 | 0.100 | 33% |
| 0.2 | 0.071 | 0.200 | 51% |
| 0.3 | 0.106 | 0.300 | 57% |

Two consequences worth expecting:

- **Rankings reorder, they do not merely slide.** Obscure titles move further
  than popular ones, so two anime sitting close together can swap places if their
  vote counts differ a lot.
- **Only the obscure are really affected.** An anime with 100,000 voters barely
  registers a change in `C`, which is correct — `C` was never doing much work
  there.

### Nothing extra needs tracking

The database already knows both things this needs, so no running tally has to be
kept during enrichment:

- **The new `C`** is the average of `score.mean` over the collection. That is an
  indexed payload field, so it is a read, not a number accumulated as records are
  written.
- **The trigger** is the collection's point count against the last recalculation's
  count.

What the database does *not* know is which `C` and `m` produced the scores it is
holding, and that has to be stored deliberately — the value, the date, and how
many anime it was computed from. Not for the arithmetic, but so that a stored
score can be explained and reproduced later, and so it is possible to tell
whether a record is stale. A score nobody can account for six months on is a
score nobody can debug.

### Mechanics, and one thing to check first

The pieces exist in `libs/qdrant_db/src/qdrant_db/client.py`:

- `scroll` — pages through points
- `update_payload` — writes the recomputed `weighted` back

**There is no aggregation call on the client.** Computing `C` therefore means
scrolling the whole collection and averaging in Python. At around 40,000 records
that is perfectly workable, but check whether the Qdrant version in use offers a
facet or aggregation API before writing the scroll loop — it would be cheaper and
would avoid pulling every payload across the wire.

The rewrite step touches every record but only its payload. Vectors are untouched
and nothing is re-embedded, so the cost is a bulk payload update rather than a
reindex.

### Recording it

Store the `C` and `m` used, with the date and sample size, alongside the setting.

### Where the score lives in the code

| Piece | Location |
| :--- | :--- |
| Model | `libs/common/src/common/models/anime.py::ScoreCalculations` |
| Proto | `protos/shared_proto/v1/anime.proto::ScoreCalculations` |
| Calculation | `libs/enrichment/.../pipeline/metadata_rules.py::merge_score` |
| `C` and `m` settings | `libs/enrichment/.../pipeline/config.py` |
| Payload index | `libs/common/src/common/config/qdrant_config.py` |

The model and the proto must change in the same commit. The contract checker
(`scripts/check_anime_model_proto_contract.py`) compares exact field-name sets and
fails on any mismatch. Regenerate `anime_pb2.py` after editing the proto.

Both `score.mean` and `score.weighted` are indexed payload fields, since both are
sortable and filterable. `statistics.anilist.scored_by` and
`statistics.kitsu.scored_by` were added at the same time, both providers having
gained vote counts.

**Proto field numbering.** `weighted` reuses field number 1, previously
`arithmetic_geometric_mean`, and `arithmetic_mean` was shortened to `mean`.
Reuse rather than reservation is safe here because
nothing has been written to a production store; on a live database this would
require `reserved 1` and a new field number instead.

### Known drawbacks

Recorded honestly; none of these are reasons to abandon the approach, but all of
them are real.

**The weighted score can never reach 10, or 0.** The imaginary votes always hold
the result slightly below it. A unanimous 10.00 reads 9.96 at 100,000 voters and 9.9996 at
ten million. In practice nothing scores 10 anyway — the highest provider mean in
our sample is 9.07 — and a printed 10.00 would more likely mean "almost nobody
voted" than "perfect". Both ends of the scale are compressed by the same amount.

**It is not on the same scale as a provider's score.** Ours reads systematically
low next to MAL's. Fine for ranking, misleading if displayed side by side — which
is why `mean` and `median` are kept.

**Some providers already do this to themselves.** MAL applies the same formula
with a batch of 50; AniSearch labels its number "Calculated Value"; AniList
documents `averageScore` as a weighted average. So a small amount of
double-correction happens. It only has a visible effect below a few hundred votes, and each site
corrects toward its own average rather than ours, so we accept it.

**Coverage moves the result.** `V` is the total across providers, so an anime on
one site with 2,000 voters is treated as weaker evidence than one on seven sites
with 2,000 each:

| sites, 2,000 voters each | total voters | weighted |
| ---: | ---: | ---: |
| 1 | 2,000 | 7.40 |
| 4 | 8,000 | 7.80 |
| 11 | 22,000 | 7.92 |

We judge this correct — 22,000 people is more evidence than 2,000 — but note
that a user with accounts on several sites is counted more than once, so pooled
votes slightly overstate the real headcount.

**Scores shift when `C` is recalculated.** Every refresh moves existing scores,
most by under 0.1 but obscure titles by the full change in `C`, and rankings can
reorder rather than slide. See
[Recalculating C and m](#recalculating-c-and-m). The alternative — freezing `C`
forever — would mean correcting toward a figure that stops describing the
database, so the drift is accepted deliberately.

**It cannot fix a skewed starting number.** See below.

### Open problem: provider offsets

Each provider sits at a systematically different level. Measured as a provider's
score minus the average of the others on the same anime, over 660 anime:

| provider | titles | offset | spread | within 0.5 of its own average |
| :--- | ---: | ---: | ---: | ---: |
| anime_planet | 648 | +0.59 | 0.65 | 67% |
| kitsu | 495 | +0.51 | 0.47 | 76% |
| mal | 654 | +0.36 | 0.43 | 82% |
| animeschedule | 446 | +0.29 | 0.63 | 61% |
| anilist | 630 | −0.27 | 0.49 | 74% |
| anisearch | 626 | −0.89 | 1.06 | 37% |
| anidb | 201 | −1.34 | 1.10 | 28% |

A **1.93 point spread**. These are not disagreements about particular anime —
AniDB's users rate everything harder than Anime-Planet's. It is a property of who
uses each site.

**The weighted score cannot repair this.** It only controls how far `M` differs
from the ordinary score; it cannot fix `M` itself. If an anime is carried only by
AniDB and AniSearch, `M` starts roughly a point too low, and with enough votes the
weighted score confidently delivers a number that is a point too low. `mean`,
`median` and `weighted` all inherit the skew.

### The offset is not a constant

It shrinks as the anime gets better:

| provider | low | mid | good | high |
| :--- | ---: | ---: | ---: | ---: |
| mal | +0.63 | +0.39 | +0.22 | +0.15 |
| anilist | −0.44 | −0.31 | −0.20 | −0.06 |
| kitsu | +1.10 | +0.64 | +0.31 | +0.08 |
| anidb | −1.91 | −1.75 | −0.79 | −0.22 |
| anime_planet | +1.20 | +0.56 | +0.31 | +0.24 |
| anisearch | −1.72 | −1.04 | −0.39 | −0.15 |
| animeschedule | +1.24 | +0.47 | −0.05 | −0.22 |
| *titles* | *150* | *231* | *197* | *82* |

The sites broadly agree on what is good and disagree on what is bad. Fitting a
straight line per provider shows why — they use different amounts of the scale:

| provider | slope |
| :--- | ---: |
| anidb | 0.47 |
| anisearch | 0.48 |
| mal | 1.09 |
| anime_planet | 1.08 |
| kitsu | 1.31 |
| animeschedule | 1.22 |

AniDB and AniSearch spread their scores over roughly twice the range of Kitsu and
AnimeSchedule. They are not simply harsher; they use the full scale while the
others compress into the upper half.

Era matters less but is visible: AniSearch runs −1.39 on pre-1990 titles and −0.15
on 2020s ones, and AniDB drifts the other way.

### Correcting it does not work

Offsets fitted on half the anime and applied to the other half, which the fit never
saw. "Truth" is the average of every provider covering that anime:

| providers carrying it | cases | no correction | constant shift | rescale |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,694 | 0.43 | 0.41 | **0.34** |
| 2 | 3,862 | 0.27 | 0.28 | 0.27 |
| 3 | 4,708 | **0.18** | 0.21 | 0.24 |

- **A constant shift is worthless.** It helps marginally at one provider and hurts
  at three, because subtracting an average overcorrects good anime and
  undercorrects bad ones.
- **A rescale helps only single-provider anime**, by about 20%. At two providers it
  is a wash, at three it is worse than doing nothing.
- **The median does not help either.** Measured separately: no better than the mean
  at one or two providers, worse at three. A median defends against one odd
  provider among many; when only two cover an anime and both use a wide scale, the
  middle of those two is still wide.

**Caveat on this test.** "Truth" is the uncorrected all-provider average, so a
subset naturally approaches it as more providers join, and any correction moves
away from it. The test is structurally biased toward doing nothing at higher
provider counts, so the k=3 result overstates the case against correction.

### Status

**Not implemented, and not recommended on current evidence.**

The only clear win is single-provider anime — which are also the titles with the
fewest votes, already pulled toward ordinary by the weighted score. The correction
would largely duplicate something we already do, in exchange for seven fitted
constants needing revalidation whenever the provider set changes.

The offsets are real and worth knowing. They do not justify a correction here.

If revisited: AniDB appears on only 201 of 660 titles, far fewer than the others,
so its −1.34 is the least certain figure in the table and it is the most extreme
provider. Any correction would rest most heavily on its weakest number.

### Rejected alternatives

| Approach | Why not |
| :--- | :--- |
| Weight providers by their voter counts | MAL takes 78% of the decision; adding providers stops mattering |
| Weight providers by square root or log of votes | Square root still gives MAL 57%; log lands within 0.016 of the plain mean, so it stores nothing new |
| Use the offline database's score | Built from a different set of sites — includes four we do not use, excludes one we do |
| Set `C` to 5 as a neutral midpoint | Not neutral; drags every thin title down by 0.63 on average |
| Hide scores below a fixed vote threshold | Needs an arbitrary number, creates a cliff at it, and still publishes nothing useful for the titles it hides |
| Use the median to defend against provider skew | Measured: no improvement at one or two providers, worse at three |
| Correct each provider's offset before merging | Measured on 660 anime, held out: a constant shift is worthless, a rescale helps only single-provider titles |

---
## Links

Three fields can hold a link to the same place, so ownership has to be explicit.

### Comparison folds the URL, not just the string

`normalize_link_url` produces a comparison key, not a URL to visit. It drops the scheme, `www.`, trailing slashes, percent-encoding and tracking parameters, and folds `x.com` → `twitter.com`. The scheme matters in practice: providers publish the same official site as both `http://` and `https://`, and keeping it filed the Toei page twice.

What it deliberately does **not** collapse:

- **Different paths on one host** — `shingeki.tv` and `shingeki.tv/season1` are different pages; `twitter.com/kimetsu_off` and `twitter.com/DemonSlayerUSA` are different accounts.
- **YouTube `/@handle` vs `/channel/<id>`** — the same channel for One Piece (`@onepieceofficial` = `channel/UCdAHaWcKdpbT5XkN2Er6BUQ`), confirmed by hand. Proving it in general needs a network lookup, and it occurs once in 487 anime; folding it moves link loss from 24% to 23%, so the dependency is not worth taking. Note One Piece still has **two** genuine YouTube channels once that pair is folded — `@onepieceofficial` (Japanese) and `@OnePieceOfficialENG` (English).

### `sources` — one URL per work

Providers decorate the same id differently. MAL publishes `/anime/21/One_Piece` where AniList cross-links `/anime/21`; AnimeSchedule links `anime-planet.com` where Anime-Planet itself says `www.anime-planet.com`. A plain string union keeps all of them: on One Piece that is 13 URLs for 7 works. Identity comes from `canonical_url_key`, and the longest spelling of each work wins — these are work pages with no query strings, so length tracks how much identity the URL states.

**Both halves are unioned because neither is complete.** The offline seed carries livechart, simkl and animecountdown, which no provider returns; the providers carry animeschedule, which the seed does not list.

AniDB is addressable two ways as well: MAL links its works through the old `anidb.net/perl-bin/animedb.pl?show=anime&aid=69` address. `canonical_url_key` recognises it, so it folds into `anidb.net/anime/69` rather than reading as a separate work — which matters for relations too, since the same resolver backs `relationship_merger`.

Kitsu is the one provider addressable two ways — providers report the slug (`kitsu.io/anime/one-piece`), the seed the numeric id (`kitsu.app/anime/12`) — and neither string reveals the other. The numeric form drops when a slug names the same work. MAL and AniList identify every work numerically, so their keys never meet a slug rival and are never dropped. Both slugged and slugless URLs are safe to feed back to the crawlers: MAL follows its own canonical link, and Kitsu resolves a slug via `filter[slug]`.

### `external_sources` is the residual

A link is excluded when another field already owns it, decided **by platform as well as by URL**. URL comparison alone is not enough: AniDB links Crunchyroll as `/series/GRMG8ZQZR` where MAL links `/series-257631`, so the URLs differ and a streaming link would sit in the residual field. The platform does not differ, and that settles it.

A claimed link is **moved, not discarded**. `merge_streaming_sources` reads from `external_sources` as well as its own field, so AniDB's Crunchyroll, Amazon and Funimation links — at URLs no other provider reports — land in `streaming_sources` instead of vanishing. Excluding them from the residual without collecting them elsewhere lost three links outright on One Piece, Funimation entirely.

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
## Synonyms

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
## Relationships

`relationship_merger.py`, called from the same pass as the metadata merge so one
call yields a complete record. Two fields:

```
related_anime:           dict[AnimeRelationType,         list[RelatedAnime]]
related_source_material: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]]
```

The goal is that one real-world work appears **exactly once, under exactly one
relation key**, carrying every source URL that mentioned it.

### The rules

- **Identity** — exact source-URL match, else normalised title. The same URL
  resolver backs this and `sources`, so AniDB's old
  `anidb.net/perl-bin/animedb.pl?show=anime&aid=69` address folds into
  `anidb.net/anime/69` here too, rather than reading as a second work.
- **Sentinels never win.** `UNKNOWN` and `OTHER` mean "no signal", so any
  concrete relation type supersedes them regardless of provider rank. They
  survive only when every provider agrees there is nothing better. This is why
  AniDB's rank does not matter for relation type: it reports `UNKNOWN` for every
  observed relation, so the guard discards it whatever its position.
- **Ties** — `PROVIDER_PRIORITY`, concrete values only.
- **`sources` and `images`** — union across providers, order-preserving.
- **Scalars** — first non-null by priority.

### Why one order for title, format and relation

`docs/anime_relationship_and_format_type_mappings.md` ranks AniDB above AniList
for format specifically. That deviation is deliberately not reproduced, because
the sentinel guard already discards AniDB's contribution to format — it answers
`UNKNOWN` — so a separate order would add a second ranking to maintain and change
nothing.

Kitsu is placed last in `PROVIDER_PRIORITY` because it supplies no relationship
data in any observed run, so there is no evidence on which to rank it higher.
That reasoning is specific to relations: it carries no authority for other
fields, and Kitsu is one of the better sources of company data.

---

## Evidence and Its Limits

Every figure in this document comes from one of these. Sample sizes are small
enough that they are design evidence, not settled constants.

| Area | Sample |
| :--- | :----- |
| Field behaviour generally | The agent directories, of which `One_agent2` through `One_agent5` each hold all seven providers. All four produce identical output, so that is **one work captured four times, not four samples** — enough to design against, not to generalise |
| [Categories](#categories) | 30 anime, all seven providers, 2,450 values |
| [Companies](#companies), folding | 684 anime, 3,398 names, all seven providers |
| [Companies](#companies), role rule | 339 appearances of companies that never animate |
| [Score](#score), weighting schemes | 7 anime, four providers |
| [Score](#score), the value of `C` | 684 anime, reweighted to the database's coverage shape |
| [Score](#score), provider offsets | 660 anime, split-half |
| [Links](#links) | 320 anime — AniList 445 links, AniDB 133, MAL 22, AniSearch 20 |
| [Synonyms](#synonyms) | 393 values from 40 anime |

### Known weaknesses

- **The weighting-scheme comparison used four providers, not seven.** AniDB,
  Anime-Planet and AnimeSchedule were absent, and two of those are the harshest
  raters. The conclusion — that vote-weighting hands MAL the decision — would only
  strengthen with them included, but the exact percentages would move.
- **AniDB is understated in every company and score measurement here.** It was
  banned throughout collection and served only what the HTTP cache already held.
- **Coverage levels were sampled at about 12 anime each** for `C`, so 6.22 is
  roughly-6.2 rather than precisely 6.22.
- **The company role rule was validated against outside knowledge deliberately** —
  companies whose type is public fact. Grading a rule against the same providers
  it consults is circular.
- **"Truth" in the provider-offset test is the uncorrected all-provider average**,
  so the test is structurally biased toward doing nothing at higher provider
  counts.

### A known defect this merge depends on

**`Success Rate: 100.0%` is reported in the same run whose summary shows
`anidb: ✗`.** `api_fetcher.py:278` computes it from
`len(api_timings) + len(api_errors)`, but a helper that returns `None` — no data,
no exception — still records a timing and never lands in `api_errors`. "Returned
nothing" therefore counts as success, so a provider silently supplying nothing is
invisible in the run summary. Every coverage figure in this document was gathered
by counting what actually arrived rather than trusting that number.
