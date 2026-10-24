# work_matching test fixtures

| File | Holds |
| -- | -- |
| `seed_entries.json` | A few anime-offline-database entries, in the seed file's own shape. |
| `canonical_records.json` | Canonical records the enrichment helpers returned, by page address. |
| `matching_cases.json` | Real seed entries with the details of their provider pages, for the matching rules. |

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
