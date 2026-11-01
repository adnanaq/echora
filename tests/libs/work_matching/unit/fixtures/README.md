# work_matching test fixtures

| File                     | Holds                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `seed_entries.json`      | A few anime-offline-database entries, in the seed file's own shape.                 |
| `canonical_records.json` | Canonical records the enrichment helpers returned, by page address.                 |
| `matching_cases.json`    | Real seed entries with the details of their provider pages, for the matching rules. |

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
