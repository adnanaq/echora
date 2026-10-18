# seed_cleaning

Finds the seed entries that need checking before enrichment, fetches every
page of them so the work matching rules can decide from their links, and
records where each page of an entry with doubled providers ends up. It reads
the downloaded seed release and never changes it. The library is deleted once
discovery replaces the seed; `libs/work_matching` stays.

Terms are as in `GLOSSARY.md` and the `work_matching` README.

## Entries to check

| Entries to check | Found by | Why |
| -- | -- | -- |
| **Doubled provider** | An entry with pages of one provider carrying more than one id. Simkl and AnimeCountdown count as one provider. | The same rule `decide_doubled_providers` uses (`find_doubled_provider_pages`), so exactly the pages it decides are checked. |
| **Split-work candidates** | Entries matching by title, type, year and season, joined into groups. | A site often adds a page before the seed joins it to the work's main entry, so the page becomes an entry of its own. |

A split-work match needs:

- the known-works title lookup to find one entry for the other: title equal once
  case, punctuation, accents and season wording are folded, same type, and
  equal years or a missing year. A missing year matches because the small entry
  of a split work usually has none yet;
- seasons equal, next to each other, or one missing. Seasons far apart are
  different works ("Gelpiyo", winter and fall); neighbouring ones go either way.

Entries joined through a shared match form one group. Then by shared provider,
the providers with pages in two or more of the group's entries:

| Shared providers | Group | Why |
| -- | -- | -- |
| none | kept | Nothing says the entries are different works. |
| one | kept, marked with that provider | One provider listing the entries apart does not decide: such groups are sometimes one work and sometimes not. |
| two or more | dropped and logged | Two providers each listing the entries apart points to different works. |

Title, type and year only find candidates; the split-work rules merge a group only when links between its
entries' pages tie them together.

## Fetching pages

`ProviderPageFetcher` fetches each page once through the public
`ApiFetcher.fetch_all_data`, one service at a time, with characters and
episodes off, and keeps it in the `ProviderPageStore`. The same fetch fills the
cache enrichment reads later.

- **Which pages:** those of the providers enrichment has a helper for: MAL,
  AniList, Kitsu, AniDB, Anime-Planet, AniSearch. AnimeSchedule is not in the
  seed: its page is looked up through each of the entry's MAL pages (else
  AniList, else AniDB) and records that page in `found_by`. One AnimeSchedule
  page found through pages of two entries keeps both lookup pages.
- **Dead pages:** a page is stored as dead only when its provider said it does
  not exist, that is when its helper answered None. A failed fetch (a timeout,
  a block, a page that cannot be read, a service that did not run, a payload
  without an anime record) is not stored, so the next run tries it again. AnimeSchedule finding nothing
  stores nothing: there is no page.
- **Reruns:** a page already stored, or an AnimeSchedule lookup that found a
  page, is not fetched again. A lookup that found nothing is repeated on every
  run, so a page AnimeSchedule adds later is still found.
- **Pace:** each provider has its own queue, one page at a time, so each
  site's rate limits hold while different sites run side by side. AnimeSchedule
  lookups wait 3 s between them: requests without an app token get its stricter,
  undocumented limit.

## Correcting entries with doubled providers

`correct_doubled_provider_entries` decides each entry with a doubled provider
from the stored pages, including the pages found through them. When the rules
settle the whole entry, its pages are placed at once; otherwise the entry is one
review case, and its places come from the kept answer once a person gives a
valid one (see "Placing an entry's pages" in the `work_matching` README). Each
placed page becomes a `SeedCorrection`:

| Field | Holds |
| -- | -- |
| `entry_reference_page` | The reference page of the seed entry that lists the page. |
| `page_key`, `address` | The page as the seed entry lists it. |
| `work` | `entry`, `other work` or `work 2`… |
| `is_main_page` | Whether enrichment fetches this page for its provider in its work. |
| `renamed_to` | The current page key, when the seed lists the page under an old address. |
| `target_entry_reference_page` | The reference page of the seed entry an `other work` page joins; `null` when it starts a new entry. |
| `reason` | The decision's reason by name, or `review answer`. |

The rules:

- **Final places, not steps.** A correction says where its page ends up, so
  the order corrections are applied in never matters, even when a page moves
  into an entry that is itself being corrected.
- **Rebuilt every run** from this run's decisions and the kept answers, so the
  file always matches them and needs no clean-up.
- **Pages without a place stay** as the seed lists them and get no correction.
- **Moves.** An `other work` page joins the seed entry its links name when they
  name exactly one; otherwise it starts a new entry. A page moves into a seed
  entry only when that entry holds no page of the same provider and its year
  and type agree with the page's (`find_move_target`). The rules and review
  answers use the same check, so neither makes a new doubled provider; a page
  an answer sends to another work without a sure target starts a new entry.
  Pages an answer puts in the same `work N` form one new entry.
- **Renamed pages.** The old address is dropped: the entry also lists the page
  under its current address, or the rules send the entry to review.
- **Main page first**, because enrichment fetches the first address of each
  provider.

`build_corrected_entries` turns the corrections into `WorkEntry`s: each seed entry
they change, without the pages that left it and with those moved in, and each
new entry. A new entry takes its title, type, year and season from its most
trusted fetched page; a page with no fetched details only has the entry it came
from to go by, so it takes that entry's title and an unknown type.

## Files

| File | Holds |
| -- | -- |
| `doubled_provider_entries.py` | `find_doubled_provider_entries`: the entries holding a doubled provider. |
| `split_work_candidates.py` | `SplitWorkCandidates` and `find_split_work_candidates`: the candidate groups. |
| `provider_page_fetcher.py` | `ProviderPageFetcher`, `FetchSummary`, `FetchOutcome`: fetching every page of the entries to check. |
| `seed_corrections.py` | `correct_doubled_provider_entries`, `SeedCorrection`, `build_corrected_entries`, `write_corrections`: where each page of an entry with doubled providers ends up. |

## Use

```bash
uv run python scripts/clean_seed.py                  # list the entries to check, fetch their pages, correct them
uv run python scripts/clean_seed.py --no-fetch       # correct them from the pages already stored
uv run python scripts/clean_seed.py path/to/seed.json --data-dir data/seed_cleaning
```

It writes to `data/seed_cleaning/` (git-ignored):

| File | Holds |
| -- | -- |
| `entries_to_check.json` | The seed's `$schema` (its address names the release), the doubled-provider entries with their doubled pages, and the candidate groups with their entries. Rewritten on each run. |
| `provider_pages.jsonl` | Every fetched page, one per line; see `ProviderPageStore`. |
| `review.jsonl` | One case per entry the rules cannot place, where a person writes the answer; see `ReviewFile`. Answers are kept across runs. |
| `corrections.jsonl` | Every `SeedCorrection`, sorted by entry reference page and page key. Rewritten on each run. |
