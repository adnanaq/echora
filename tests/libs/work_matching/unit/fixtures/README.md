# work_matching test fixtures

| File                     | Holds                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `seed_entries.json`      | A few anime-offline-database entries, in the seed file's own shape.                 |
| `canonical_records.json` | Canonical records the enrichment helpers returned, by page address.                 |
| `matching_cases.json`    | Real seed entries with the details of their provider pages, for the matching rules. |
| `review_cases.jsonl`     | Real lines of seed cleaning's review file, for the review page's choices and saves. |

## matching_cases.json

Each case is named after the situation it shows (`puu_neko` for a mixed entry,
`cocoon` for a renamed page) and holds:

- `entries`: the seed entries, with only the fields work matching reads
  (`sources`, `title`, `type`, `animeSeason`);
- `pages`: one `ProviderPage` per fetched page, as `ProviderPage.model_dump()`
  writes it, including searched pages such as AnimeSchedule's.

The pages were fetched through the enrichment helpers, with characters and
episodes off, and built with `ProviderPage.from_canonical_record`;
AnimeSchedule pages were searched by the entry's MAL page and carry it as
`searched_by`. To refresh a case, fetch its entries' pages the same way and
replace its `pages`, then check the tests' expected decisions against the new
pages before changing them.

Cases taken from the whole-seed provider page store (pages fetched for every
entry of seed release `2026-40`), on 2026-10-25:

| Case               | Shows                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------- |
| `douluo_dalu_arcs` | Arc entries whose doubled Kitsu pages are parts of MAL's _Douluo Dalu 2nd Season_.     |
| `shin_oishinbo`    | A group only partly tied by links.                                                     |
| `ajin_films`       | AniDB's single page for three films MAL lists apart, linked from each film's MAL page. |

## review_cases.jsonl

Four lines copied unchanged from `data/seed_cleaning/review.jsonl`, written by
`scripts/clean_seed.py` on seed release `2026-43` (2026-11-01), so saves can be
checked byte for byte. The `review_cases` fixture names them:

| Name                | Case                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| `atashinchi_next`   | Split work: two entries of _Atashin'chi Next 2nd Season_ tied by an AnimeSchedule page.        |
| `gochuumon_movie`   | Doubled provider: two AniDB pages in _Gochuumon wa Usagi desu ka? Movie_, one of another work. |
| `nami`              | Linked page: an AniList page whose type differs from its Kitsu entry's.                        |
| `aware_meisaku_kun` | Split work: seven entries, 46 pages, with no suggestion.                                       |

To refresh them, copy the lines with the same case names from a newer run, and
check the tests' expected choices against them before changing the tests.
