# seed_cleaning

Finds the seed entries that need checking before enrichment, and fetches every
page of them so the work matching rules can decide from their links. It reads
the downloaded seed release and never changes it. The library is deleted once
discovery replaces the seed; `libs/work_matching` stays.

Terms are as in `GLOSSARY.md` and the `work_matching` README.

## Entries to check

| Entries to check | Found by | Why |
| -- | -- | -- |
| **Doubled provider** | An entry with pages of one provider carrying more than one id. Simkl and AnimeCountdown count as one provider. | The same rule `decide_doubled_providers` uses (`doubled_provider_pages`), so exactly the pages it decides are checked. |
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

## Files

| File | Holds |
| -- | -- |
| `doubled_provider_entries.py` | `doubled_provider_entries`: the entries holding a doubled provider. |
| `split_work_candidates.py` | `SplitWorkCandidates` and `split_work_candidates`: the candidate groups. |
| `provider_page_fetcher.py` | `ProviderPageFetcher`, `FetchSummary`, `FetchOutcome`: fetching every page of the entries to check. |

## Use

```bash
uv run python scripts/clean_seed.py                  # list the entries to check and fetch their pages
uv run python scripts/clean_seed.py --no-fetch       # only list the entries to check
uv run python scripts/clean_seed.py path/to/seed.json --data-dir data/seed_cleaning
```

It writes to `data/seed_cleaning/` (git-ignored):

| File | Holds |
| -- | -- |
| `entries_to_check.json` | The seed's `$schema` (its address names the release), the doubled-provider entries with their doubled pages, and the candidate groups with their entries. Rewritten on each run. |
| `provider_pages.jsonl` | Every fetched page, one per line; see `ProviderPageStore`. |
