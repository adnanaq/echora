# Score Calculation

**Last updated**: 2026-09-24
**Purpose**: How we turn seven providers' scores into our own, why the formula is
shaped the way it is, and what it does badly.

Our score is not any provider's score. Each site publishes a number built from
its own users, on its own scale, with its own habits. We publish three numbers
derived from all of them. This document records how those are calculated and,
more importantly, **why** — so that a constant which looks arbitrary is not
"tidied up" into a bug, and so that nobody re-runs measurements already done.

---

## Table of Contents

1. [What We Publish](#what-we-publish)
2. [The Weighted Score](#the-weighted-score)
3. [Why Not Weight The Providers Themselves](#why-not-weight-the-providers-themselves)
4. [Why C Is 6.2](#why-c-is-62)
5. [Why m Is 1000](#why-m-is-1000)
6. [When We Publish Nothing](#when-we-publish-nothing)
7. [Recalculating C and m](#recalculating-c-and-m) — **not built yet**
8. [Where It Lives In The Code](#where-it-lives-in-the-code)
9. [Known Drawbacks](#known-drawbacks)
10. [Open Problem: Provider Offsets](#open-problem-provider-offsets)
11. [Rejected Alternatives](#rejected-alternatives)
12. [Evidence and Its Limits](#evidence-and-its-limits)

---

## What We Publish

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

---

## The Weighted Score

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

---

## Why Not Weight The Providers Themselves

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

---

## Why C Is 6.2

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

---

## Why m Is 1000

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

---

## When We Publish Nothing

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

---

## Recalculating C and m

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

---

## Where It Lives In The Code

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

---

## Known Drawbacks

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

---

## Open Problem: Provider Offsets

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

---

## Rejected Alternatives

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

## Evidence and Its Limits

Every figure above comes from one of these. Sample sizes are small enough that
they should be treated as design evidence, not as settled constants.

| Measurement | Sample | Method |
| :--- | :--- | :--- |
| Provider weight shares and scheme comparison | 7 anime, 4 providers (MAL, AniList, Kitsu, AniSearch) | Direct API and page fetches |
| `C` from our providers | 110 anime, stratified across coverage levels, reweighted to the database's real distribution | Full `ApiFetcher` run |
| Movement caused by the correction | 93 anime (110 minus those with no score or no votes) | Same run |
| Agreement versus voter count | 110 anime, all seven providers | Same run |
| Provider offsets, by score band and era | 660 anime, stratified across five eras and all formats | Full `ApiFetcher` run, 739 fetched, 0 timeouts |
| Offset correction benefit | 660 anime | Split-half: fitted on one half, tested on the other |


### Known weaknesses in this evidence

- **The scheme comparison used four providers, not seven.** AniDB,
  Anime-Planet and AnimeSchedule were absent, and two of those are the harsh
  raters. The conclusion (vote-weighting hands MAL the decision) would only
  strengthen with them included, but the exact percentages would move.
- **The all-provider run covered 110 of an intended 120 anime.** It stalled on a
  browser crawl caught by a Cloudflare challenge and was stopped; results are
  written incrementally, so the 110 are complete.
- **Coverage levels were sampled at about 12 anime each.** The measurement gave
  6.22; the setting is 6.2, because the second decimal is not supported by that
  sample size.
- **AniDB covers only 201 of the 660 offset titles.** Its −1.34 and its 0.47 slope
  rest on a third of the data behind the other providers, and it is the most
  extreme provider in the table.
- **The 660 include 144 anime selected under an earlier sampling rule** that
  required the entry to carry an offline-database score. That rule was dropped;
  the titles themselves are valid measurements, but the set is not uniformly
  sampled.
- **"Truth" in the offset test is the average of all covering providers**, which
  itself contains the offsets. The test therefore measures whether sparse titles
  agree with well-covered ones — consistency, not correctness. If every site is
  skewed the same way, nothing here would detect it.

### Superseded claims

Kept so nobody re-derives them from stale notes:

- **Correcting provider offsets was recommended on the strength of a 25% error
  reduction.** That came from 45 anime, leave-one-out. At 361 anime the benefit had
  gone; at 660, split-half, a constant shift is worthless and a rescale helps only
  single-provider titles. The recommendation was withdrawn.
- An early figure said the correction moves thin titles by **0.56** on average.
  That came from a test using only MAL, AniList and Kitsu, which excluded most
  sparse titles for lacking an AniList link. With all seven providers the median
  movement is 0.25, but 48% of titles move more than 0.25 and 11% move more than
  1.00 — the effect is narrower in the middle and much larger in the tail than
  that first number suggested.
- A follow-up claimed the correction "barely moves sparse titles, 0.20". Same
  cause, same correction: three providers, thin titles missing.
- It was claimed that an anime scoring exactly `C` being left unchanged shows `C`
  is set sensibly. It does not — that holds for any value of `C`. What matters is
  *which* score is left unchanged, and it should be the ordinary one.
