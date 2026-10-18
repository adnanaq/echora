# work_matching

Decides which provider pages describe the same anime work. Seed cleaning uses it
now to repair the seed; seed link filling and discovery will use it later.

The library only decides. It does not fetch pages, and it does not change the
seed: the caller fetches the pages, and seed cleaning applies the decisions. It
imports nothing from `enrichment`.

## Terms

| Term | Meaning |
| -- | -- |
| **Work** | One real anime title as a whole: a single TV season, movie, OVA or special. A sequel season is a separate work. |
| **Provider** | A site that catalogues anime: MAL, AniList, Kitsu, AniDB, Anime-Planet, AniSearch, ANN, LiveChart, Simkl, AnimeCountdown. AnimeSchedule is fetched too, but the seed does not list it. |
| **Provider page** | One page on a provider about one work, such as `https://myanimelist.net/anime/21`. |
| **Page key** | The name of a provider page, whatever the spelling of its address: `mal:anime:21` for `https://www.myanimelist.net/anime/21/One_Piece`. Pages are always compared by key. |
| **Seed** | The downloaded anime-offline-database file the enrichment pipeline starts from. |
| **Seed entry** | One row of the seed: a title, type, year, season and a list of provider pages. |
| **Work entry** | One work as this library sees it (`WorkEntry`): title, type, year, season and its pages by key. Built from a seed entry, later also from discovery. |
| **Known works** | All work entries, looked up by page key or by title, type and year (`KnownWorks`, backed by the seed in `KnownWorksFromSeed`). The seed lookup also reports pages listed in several entries and entries without pages. |
| **Linked** | Two pages are linked when either one lists the other among its links to other providers. Direction does not matter. |
| **Doubled provider** | An entry listing two or more pages of one provider. It may be one page under two addresses, one work the provider lists twice, or two works mixed into one entry. Checking tells which. |
| **Mixed entry** | An entry whose pages cover more than one work. Puu-Neko holds the pages of both "Puu-Neko" and "Puu-Neko Shougekijou". |
| **Duplicate page** | A provider having more than one page for the same work, such as Kitsu's "Yume Mitsuya" and "Yume Juuya" pages for one short. |
| **Split work** | One work spread over several seed entries, such as "CUE!" (only its ANN page) and "Cue!" (every other page). |
| **Renamed page** | A page whose address changed; the old address redirects to the new one. Anime-Planet renames pages. The old and new address are one page. |
| **Found page** | A page found through another provider's page rather than listed in the seed. AnimeSchedule is looked up by MAL, AniList or AniDB id. |
| **Lookup page** | A page a found page was found through. `ProviderPage.found_by` lists every one: a found page can be reached through pages of several entries, such as both entries of a split work. |
| **Dead page** | A page the provider no longer serves (`ProviderPage.is_dead`). |
| **Multi-work page** | A page naming two or more pages of one provider, such as an anthology page linking several MAL pages. Its links prove nothing. |
| **Review** | A decision the rules do not settle. A person (later the review agent) answers it. |

## Files

| File | Holds |
| -- | -- |
| `provider_page_key.py` | `get_page_key`: an address to its page key. `get_page_provider`: a key's provider. `rank_by_provider_trust`: how far a page's provider is trusted to name a work, in `REFERENCE_ORDER`. |
| `work_entry.py` | `WorkEntry`, and the type and year comparisons `is_type_matching` and `is_year_matching`. |
| `comparable_title.py` | `fold_title_for_comparison`: a title folded for comparison (case, punctuation, Latin accents, "2nd Season"). |
| `known_works.py`, `known_works_from_seed.py` | The `KnownWorks` interface and its seed-backed lookup. |
| `provider_page.py` | `ProviderPage`: what a provider says about one page, read from the canonical record its enrichment helper returns. `index_pages_by_current_key`: pages by the current key each resolves to. |
| `provider_page_store.py` | `ProviderPageStore`: fetched pages kept in a JSON Lines file, so they outlive the 7-day HTTP cache. `has_page_found_by` tells whether a lookup through a page, such as AnimeSchedule's, was done; `get_pages_found_by` returns the pages it found. |
| `provider_page_links.py` | `ProviderPageLinks`: which pages of a group are linked. |
| `doubled_provider_check.py` | `decide_doubled_providers`: the decision for each doubled provider of an entry. `find_doubled_provider_pages`: an entry's doubled providers and their pages, the rule seed cleaning lists entries by. `get_decided_provider`: the provider a page's decision falls under, AnimeCountdown under Simkl. `get_seed_type`: the seed type a provider page's type falls under. |
| `split_work_check.py` | `decide_split_work`: whether separate entries are one split work. |
| `page_placement.py` | `PagePlacement`: where one page of an entry with doubled providers goes. `place_pages_without_review`: the places when the rules settle the whole entry. `read_answer_placements`: an answer read into places. `suggest_page_placements`: the places the links point to, offered to a reviewer. |
| `review_cases.py` | `ReviewCase`: one entry or split-work group left to a person, with its answer. `ReviewFile`: the review file, its answers kept across runs. |
| `file_replacement.py` | `replace_file_lines`: a file's lines replaced in one step, so an interrupted run leaves the old file. |

## The flow

```mermaid
flowchart TD
    seed[Seed file] --> entries[WorkEntry per seed entry,<br/>pages by page key]
    seed --> known[KnownWorksFromSeed:<br/>every entry by page and by title]
    entries --> fetch[Caller fetches every page<br/>with the enrichment helpers]
    fetch --> store[(ProviderPageStore)]
    store --> pages[ProviderPage per page:<br/>titles, type, year, season, links]
    pages --> links[ProviderPageLinks:<br/>links between the pages,<br/>in both directions]
    links --> doubled[decide_doubled_providers:<br/>each doubled provider of an entry]
    links --> split[decide_split_work:<br/>candidate entries]
    known --> doubled
    doubled --> placed{place_pages_without_review:<br/>whole entry placed?}
    placed -- yes --> caller[Caller applies the places:<br/>seed cleaning writes corrections]
    placed -- no --> review[(ReviewFile:<br/>one review case per entry<br/>or split-work group)]
    split --> settled{Settled?}
    settled -- yes --> caller
    settled -- no --> review
    review -- "a person answers;<br/>the next run hands the answer back" --> caller
```

1. **Read the seed.** Each seed entry becomes a `WorkEntry`, its pages keyed by page key. `KnownWorksFromSeed` indexes every entry by page and by title.
2. **Fetch the pages.** The caller fetches each page through the enrichment helpers, with characters and episodes off, and builds a `ProviderPage` with `ProviderPage.from_canonical_record`. AnimeSchedule is looked up through the entry's MAL page (else AniList, else AniDB) and records that page as `found_by`. Pages are kept in a `ProviderPageStore`.
3. **Link the pages.** `ProviderPageLinks` joins the candidate pages by their links, in both directions:
   * a renamed page resolves to its current key;
   * a found page counts as each of its lookup pages, so AnimeSchedule's links count for every page it was found through, and its link back to that page is ignored;
   * a link to a page of the same provider proves nothing and is ignored;
   * the links of a multi-work page are set aside.
4. **Decide.** `decide_doubled_providers` decides each doubled provider of an entry, and `decide_split_work` decides whether candidate entries are one work.
5. **Place or review.** `place_pages_without_review` places every doubled page of an entry when the rules settle the whole entry, and the caller applies the places. Otherwise the entry becomes one review case in the `ReviewFile`, as does a split-work group the links do not settle; once a person answers, the next run hands the answer to the caller.

Only links decide. A page's title, year, type, season and release date never
decide: a provider listing one work twice gives the two pages different titles
("Ankeeto" and "The Question" on Anime-Planet), and picking one by title drops
the other. These details are recorded on review cases to help the reviewer.

## Deciding a doubled provider

A doubled provider is a provider with two or more pages in one entry.
`decide_doubled_providers` first gives each of those pages one of three labels,
from its links alone:

| Label | When |
| -- | -- |
| **Belongs** | The page is linked with a page of another provider in this entry, in either direction. |
| **Unrelated** | The page links a page outside this entry that names a different work: a page of a provider this entry already has (so the two disagree), or a page of another known work. |
| **Undecided** | Neither: no link ties the page to this entry or to another work. Pages of providers that are not fetched (ANN, LiveChart, Simkl, AnimeCountdown) end here unless a fetched page links them. |

Before the labels count, two checks can send the whole provider to review. A
dead page has no links to read, and a link from a multi-work page does not say
which work it means; either could change which pages belong, so none of the
provider's pages is decided.

The checks run in this order, and the first that applies gives the reason:

```mermaid
flowchart TD
    start[The provider's pages in the entry] --> same{Do all the addresses name one page,<br/>once renamed pages resolve<br/>to their current key?}
    same -- yes --> SAME_PAGE[SAME_PAGE<br/>outcome: same page]
    same -- no --> multi{Does a multi-work page<br/>link one of the pages?}
    multi -- yes --> MULTI_WORK_LINK[MULTI_WORK_LINK<br/>review]
    multi -- no --> dead{Is one of the pages dead?}
    dead -- yes --> DEAD_PAGE[DEAD_PAGE<br/>review]
    dead -- no --> both{Is a page linked with the entry<br/>and also linking another work?}
    both -- yes --> CONTRADICTING_LINKS[CONTRADICTING_LINKS<br/>review]
    both -- no --> undecided{Is a page undecided?}
    undecided -- yes --> UNDECIDED_PAGES[UNDECIDED_PAGES<br/>review]
    undecided -- no --> count{How many pages belong?}
    count -- none --> NONE_LINKED[NONE_LINKED<br/>review]
    count -- one --> ONE_LINKED[ONE_LINKED<br/>outcome: one work]
    count -- several --> shared{Are they all linked<br/>through one same page?}
    shared -- "yes, and their years<br/>and types agree" --> SHARED_LINK[SHARED_LINK<br/>outcome: one work]
    shared -- "yes, but their years<br/>or types differ" --> DETAILS_DIFFER[DETAILS_DIFFER<br/>review]
    shared -- no --> apart{Are they linked through<br/>entirely separate pages?}
    apart -- yes --> LINKED_APART[LINKED_APART<br/>outcome: mixed entry]
    apart -- no --> OVERLAPPING_LINKS[OVERLAPPING_LINKS<br/>review]
```

### Reasons

| Reason | Outcome | What the links show | What happens | Example |
| -- | -- | -- | -- | -- |
| `SAME_PAGE` | same page | The addresses are one page: an old address that now leads to the page's renamed address. | Nothing is doubled; the current address is kept. | Cocoon on Anime-Planet: `cocoon-aru-natsu-no-shoujo-tachi-yori` now leads to `cocoon-one-summer-of-girlhood`. |
| `ONE_LINKED` | one work | Exactly one page belongs. Every other page links a different work. | The linked page stays. The others leave the entry, and move to the other work's entry when that work is in the seed. | Buriki no Dance: Kitsu 48612 belongs; Kitsu 51023 links MAL 64943, which is the "Buriki no Dance (2026)" entry. |
| `SHARED_LINK` | one work | Two or more pages belong, all linked through the same page of another provider, and they have the same year and type. | All of them stay: the provider lists this one work more than once, or splits into parts what another provider keeps as one. | Ao Haru Ride OVA: AniList 20837 and 20900 both link MAL 24151, which covers both episodes. |
| `LINKED_APART` | mixed entry | Two or more pages belong, but each through different pages of other providers, with no page in common. | Each page is a separate work, together with the pages it is linked with; the entry mixes them and is split along the links. | Puu-Neko: MAL 10459 goes with AniList 10459 and Kitsu 6190; MAL 11589 with AniList 11589 and Kitsu 6543. |
| `MULTI_WORK_LINK` | review | A page that names two or more pages of one provider, such as an anthology, links one of these pages. Such a link does not say which work it means. | A person decides. | An AniDB page for an anthology that links the MAL pages of two of its parts. |
| `DEAD_PAGE` | review | One of the pages no longer exists on the provider's site, so its links cannot be read. | A person decides. | |
| `CONTRADICTING_LINKS` | review | One page is linked with the entry and also links a different work, so its links disagree. | A person decides. | |
| `UNDECIDED_PAGES` | review | At least one page has no link to the entry and no link to another work, so nothing says where it belongs. | A person decides. This is the most common review: ANN, LiveChart, Simkl, AnimeCountdown and Anime-Planet pages are rarely linked. | Ao Haru Ride OVA: Anime-Planet's "Unwritten" page. |
| `DETAILS_DIFFER` | review | The pages are linked through the same page, as for `SHARED_LINK`, but their years or types differ, so one of the links may be a provider's mistake. | A person decides. | Pocari Sweat: Kitsu's 2019 ad and its 2020 ad both link AniList 139182, the 2020 ad. |
| `OVERLAPPING_LINKS` | review | The pages are linked through some of the same pages, but no single page is linked with all of them, so they are neither clearly one work nor clearly separate works. | A person decides. | |
| `NONE_LINKED` | review | Every page links a different work and none is linked with the entry, so the entry would keep none of this provider's pages. | A person decides. | |

Simkl and AnimeCountdown number a work the same, so their pages are decided
together, under the provider name `simkl`. A Simkl page and an AnimeCountdown
page with different numbers count as a doubling.

### Decision fields

`decide_doubled_providers` returns one `DoubledProviderDecision` per doubled
provider:

| Field | Holds |
| -- | -- |
| `provider` | The provider decided; `simkl` for Simkl and AnimeCountdown together. |
| `pages` | The provider's page keys in the entry, renamed pages folded into their current key. |
| `reason` | The `Reason` above. |
| `outcome` | The `Outcome` the reason gives: `SAME_PAGE`, `ONE_WORK`, `MIXED_ENTRY` or `REVIEW`. Everything but `REVIEW` can be applied as it stands. |
| `belonging_pages` | Each page that belongs, with the entry's pages linked with it. On a mixed entry these sets show which pages go together. |
| `unrelated_pages` | Each unrelated page, as an `UnrelatedPage`: its `links` naming the other work, and `other_works`, the known works those links belong to. |
| `undecided_pages` | The pages with no links either way. |
| `title_matches` | On review only: the pages whose main title matches the entry's title or the main title of one of its other pages. |
| `year_type_matches` | On review only: the pages whose year and type agree with the entry's. |
| `season_matches` | On review only: the pages in the entry's season. |

`belonging_pages`, `unrelated_pages` and `undecided_pages` are filled for every reason except
`SAME_PAGE`. On a review they show what the links say, even when a check that
runs first, such as `DEAD_PAGE`, decided the reason.

The three match fields are hints for the reviewer, never the answer: a provider
listing one work twice gives the two pages different titles, so a title match
can point at one page and leave out the other.

### Worked examples

Keys are written short: `mal:1` for `mal:anime:1`. An arrow means "lists a link
to"; links count both ways. A page missing from a table lists no links.

**1. A mixed entry.** One entry holds two MAL pages, two AniList pages and a Kitsu page.

| Page | Links |
| -- | -- |
| `anilist:1` | → `mal:1` |
| `anilist:2` | → `mal:2` |
| `kitsu:1` | → `mal:1`, `anilist:1` |

For MAL: no page is renamed, and there is no multi-work or dead page, so each
page gets its label. `mal:1` is linked with `anilist:1` and `kitsu:1`; `mal:2` with
`anilist:2`. Both belong. They share no linked page, and their linked pages
never overlap, so the reason is `LINKED_APART`: a **mixed entry**. AniList comes
out the same way.

```text
provider   reason        outcome      belonging
mal        LINKED_APART  MIXED_ENTRY  mal:1 -> {anilist:1, kitsu:1}
                                      mal:2 -> {anilist:2}
anilist    LINKED_APART  MIXED_ENTRY  anilist:1 -> {kitsu:1, mal:1}
                                      anilist:2 -> {mal:2}
```

The `belonging_pages` sets show the two works: `mal:1`, `anilist:1` and `kitsu:1`
are one; `mal:2` and `anilist:2` the other.

**2. One provider settled, another sent to review.** The entry holds `mal:3`,
two Kitsu pages and two Anime-Planet pages. Another known work has `mal:9`.

| Page | Links | Title, type, year |
| -- | -- | -- |
| `kitsu:3` | → `mal:3` | |
| `kitsu:4` | → `mal:9` | |
| `animeplanet:example` | none | Example, TV, 2020 (as the entry) |
| `animeplanet:example-movie` | none | Example Movie, MOVIE, 2020 |

For Kitsu: `kitsu:3` belongs. `kitsu:4` links `mal:9`, a MAL page the entry
does not have, so it is unrelated, and `mal:9` belongs to the known work "Other
Example". One page belongs and the other links another work: `ONE_LINKED`, **one
work**. `kitsu:4` can move to "Other Example".

For Anime-Planet: neither page has links either way, so both are undecided and
the reason is `UNDECIDED_PAGES`: **review**. The match fields point the
reviewer at `animeplanet:example`, but do not decide.

```text
provider     reason           outcome   result
kitsu        ONE_LINKED       ONE_WORK  belonging: kitsu:3 -> {mal:3}
                                        unrelated_pages: kitsu:4 -> links {mal:9}, other work "Other Example"
animeplanet  UNDECIDED_PAGES  REVIEW    undecided: animeplanet:example, animeplanet:example-movie
                                        title_matches: animeplanet:example
                                        year_type_matches: animeplanet:example
```

**3. Two pages linked with one same page.** The entry holds `mal:5`, `anilist:5`
and `anilist:6`, and both AniList pages link `mal:5`.

* Both pages are TV from 2020: they share a linked page and their year and type
  agree, so the reason is `SHARED_LINK`, **one work**. AniList lists the work
  twice, or splits what MAL files as one; both pages belong.
* `anilist:6` is from 2021 instead: they still share `mal:5`, but the year
  differs, so one link may be a provider's mistake. The reason is
  `DETAILS_DIFFER`: **review**.

## Deciding a split work

A split work is one work spread over several seed entries. `decide_split_work`
takes candidate entries, such as entries sharing a title, and returns a
`SplitWorkDecision`. The checks run in this order, and the first that fails
gives the reason:

| Reason | One work? | What it means |
| -- | -- | -- |
| `TOO_FEW_ENTRIES` | no | Fewer than two entries were given, so there is nothing to join. |
| `NOT_LINKED` | no | Links do not connect every entry: starting from the first entry and following links between their pages, at least one entry is never reached. A page that two entries both list does not connect them, since it may be wrongly added to one of them. |
| `TYPE_DIFFERS` | no | Links connect every entry, but their seed types differ. `UNKNOWN` agrees with any type. |
| `YEAR_DIFFERS` | no | Links connect every entry and the types agree, but their years differ. A missing year agrees with any year. |
| `LINKED` | yes | Links connect every entry, and their types and years agree: the entries are one work. |

Every reason but `LINKED` goes to review. A shared title alone is never enough:
remakes, music videos and short films reuse titles.

## Placing an entry's pages

A decision says which of one provider's pages belong to the entry; correcting
the entry needs a place for every page. `page_placement` gives each page a
place in the words a review answer uses, so places from the rules and from
answers are applied the same way:

| Place | Means | Applied as |
| -- | -- | -- |
| `entry` | The page belongs to this entry. | It stays. |
| `other work` | The page belongs to a work outside this entry. | It joins the seed entry its links name when they name exactly one; otherwise it starts a new entry of its own, so two `other work` pages become two entries. Give pages of one new work the same `work N` instead. |
| `work 2`, `work 3`… | The entry is mixed, and the page belongs to its second, third… work. | Pages with the same number form one new entry. One case covers the whole entry, so a number names the same work on every page. |
| `, main` after `entry` or `work N` | The page enrichment fetches when a work keeps two or more pages of one provider. | It is listed first, since enrichment takes the first address of each provider. |

**Settled without a person.** `place_pages_without_review` places the entry only
when every decision of it is one of these:

| Decision | Places | Why it is enough |
| -- | -- | -- |
| `SAME_PAGE`, the entry listing the current address | The current page `entry`; the old address is dropped. | Both addresses are one page. |
| `ONE_LINKED`, each other page naming exactly one seed entry whose year and type agree with the page's | The linked page `entry`, the others `other work`, joining that seed entry. | Each page has exactly one place, and its own details do not contradict it. |

Every other entry goes to review as one case:

* a work keeping two pages of one provider (`SHARED_LINK`) needs a person to
  name its main page;
* a mixed entry (`LINKED_APART`) needs every page placed, and some pages, often
  Anime-Planet, AniSearch, LiveChart or Simkl ones, are linked with neither
  work;
* a page naming two other works has no single place;
* a page whose links name no seed entry: the seed holds every live MAL page, so
  such a link most often comes from a stale mapping to a page the provider has
  deleted, kept by a duplicate page of the entry's own work (Kitsu's "Qian
  Qiu" maps to a deleted MAL page but is the entry's "Thousand Autumns");
* a page whose year or type disagrees with the seed entry its links name: the
  mapping itself may be wrong (Kitsu maps unaired Ganbare! Kickers episodes
  from 2005 to a 1987 recap OVA). Details only stop a move; they never place a
  page;
* a renamed page whose current address the entry does not list has no address
  to keep.

**A valid answer** places every doubled page. One that splits the entry (any
`work N`) places every page of the entry, since each must go to one of the
works; otherwise a page left out stays in the entry. In each work, a provider
with two or more pages has exactly one main page, and no other page is main.
Simkl and AnimeCountdown count as two providers here: enrichment fetches
neither.

**A suggested answer** comes with each case. Pages joined by the decisions'
links form works: the work holding the most trusted page (`rank_by_provider_trust`)
is `entry`, the others `work 2`, `work 3`… by trust. A work joining two pages
that a decision found linked apart is left out, since its pages are linked
with both works (Puu-Neko's AniDB 8328). Pages linked with nothing get no
suggestion.

## Review cases

Every entry the rules cannot place, and every split-work group they do not
settle, becomes a `ReviewCase`, one JSON line in the review file. All of an
entry's doubled providers are one case, since telling which works a mixed entry
holds needs all its pages at once. A person, or later the review agent, writes
the answer into the line; the next run hands it back.

```json
{"case": "doubled provider: animeplanet:anime:example animeplanet:anime:example-movie in mal:anime:3",
 "answer": {"animeplanet:anime:example": "entry", "animeplanet:anime:example-movie": "other work"},
 "suggested_answer": {"animeplanet:anime:example": "entry", "mal:anime:3": "entry"},
 "reasons": ["UNDECIDED_PAGES"], "entries": {"mal:anime:3": {...}}, "decisions": [...], "pages": [...]}
```

(One line in the file; wrapped here.)

| Field | Holds |
| -- | -- |
| `case` | The case's name: its kind and its sorted page keys. `doubled provider: ` with every doubled page of the entry and `in` the entry's reference page, since the answer places pages relative to that entry; or `split work: ` with the entries' reference pages. |
| `answer` | `null` until answered. |
| `suggested_answer` | For a doubled-provider case, the places the links point to (see "Placing an entry's pages"); empty for a split work. |
| `reasons` | Each decision's reason by name, so answers can be counted per reason even after its wording changes. |
| `entries` | Each entry involved, by its reference page, as its `WorkEntry`: title, type, year, season and pages. |
| `decisions` | Every `DoubledProviderDecision` of the entry, or the one `SplitWorkDecision`, as the rules made them. Sets of page keys are written sorted, so two runs give the same file. |
| `pages` | The details of the entries' fetched pages and of the pages found through them (such as AnimeSchedule), as `ProviderPage` stores them: every link the decision read. |

**Answers.** A doubled-provider answer places pages with the words in "Placing
an entry's pages"; `accepted_answer` returns it read into `PagePlacement`s. A
split work's answer is `one work` or `separate works`. An answer that is not
valid, such as a missing page, a misspelt word or two main pages, leaves the
case open and the text in place to correct.

**Across runs.** `ReviewFile` reads only each line's `case` and `answer`.
`write_cases` writes this run's cases with their kept answers, and replaces the
file in one step, so an interrupted run leaves the old file:

* a case named the same as before gets its kept answer back;
* a case whose pages changed, or whose doubled pages moved to another entry, has
  a new name, so it is asked again;
* an answer whose case did not come up is kept as written, never applied, and
  returned, so it can be reported;
* an unanswered case that did not come up is dropped.

Only `answer` is kept from a line; anything else written in it is rebuilt. An
answer that is not accepted, or a case answered on two lines, is logged as a
warning. A line that is not valid JSON stops the run, with the file and line
named, so a person's edit is never overwritten unread. Two cases with the same
name in one run raise `DuplicateCaseError`. The file is replaced through
`replace_file_lines`: written to disk next to the old one and renamed over it,
keeping its permissions; the folder is then synced, so the rename survives a
power cut. Do not edit the file
while a run is writing it: an answer saved during the run can be lost.

Applying answers is the caller's job: seed cleaning turns them into corrections.

## Use

```python
from work_matching import (
    KnownWorksFromSeed,
    Outcome,
    ProviderPageStore,
    decide_doubled_providers,
)

known_works = KnownWorksFromSeed.from_file("assets/seed_data/anime-offline-database.json")
store = ProviderPageStore("data/seed_cleaning/provider_pages.jsonl")

entry = known_works.find_by_page_key("mal:anime:10459")
pages = [page for key in entry.pages if (page := store.get_page(key))] + [
    found for key in entry.pages for found in store.get_pages_found_by(key)
]
for decision in decide_doubled_providers(entry, pages, known_works):
    if decision.outcome is Outcome.REVIEW:
        print(decision.provider, decision.reason, decision.undecided_pages)
```

Pass the found pages too (such as the entry's AnimeSchedule pages): their links
count for their lookup pages.

An entry the rules cannot place goes to the review file as one case, and its
kept answer comes back:

```python
from work_matching import ReviewCase, ReviewFile, place_pages_without_review

review = ReviewFile("data/seed_cleaning/review.jsonl")
decisions = decide_doubled_providers(entry, pages, known_works)
placements = place_pages_without_review(entry, decisions, pages)
if placements is None:
    case = review.attach_kept_answer(
        ReviewCase.from_doubled_provider_entry(entry, decisions, pages)
    )
    placements = case.accepted_answer  # None while the case is open
    unused = review.write_cases([case])
```
