# Source API Field Mappings — Verified Values

**Last updated**: 2026-10-11
**Method**: Live API calls + schema introspection against diverse anime

| Source       | Verification method                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| MAL          | MAL's own page strings, as the MAL crawlers read them (One Piece, AoT, Fate, Bleach, DB Kai, etc.)            |
| AniList      | GraphQL `__type` introspection + live queries (One Piece, AoT, Fate, Railgun, NGE, etc.)                      |
| Kitsu        | REST API calls + [server source](https://github.com/hummingbird-me/kitsu-server) (One Piece, AoT, Fate, etc.) |
| AnimeSchedule | REST API v3 calls + filter queries (One Piece, AoT, Fate/kaleid, Cyberpunk, Isekai Quartet, etc.)             |
| AniDB        | HTTP XML API calls + [wiki](https://wiki.anidb.net/Anime_Type) (One Piece, AoT, NGE, Bleach, FMA:B, etc.)     |
| AnimePlanet  | crawl4ai scraping + stored data (One Piece, AoT, Gintama, Cowboy Bebop, Shelter, Cyberpunk, etc.)             |
| AniSearch    | crawl4ai scraping (One Piece, Bleach, Naruto, DBZ, Gintama, AoT, FMA:B, Cowboy Bebop, Steins;Gate, etc.)      |

---

## Relation Types

| Canonical             | MAL                     | AniList       | Kitsu                 | AnimeSchedule   | AniDB                        | AnimePlanet                              | AniSearch                                                     | Verified with                                         |
| --------------------- | ----------------------- | ------------- | --------------------- | -------------- | ---------------------------- | ---------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| `SEQUEL`              | `"Sequel"`              | `SEQUEL`      | `sequel`              | `sequels`      | `"Sequel"`                   | subtype `"Sequel"`                       | `"Sequel"`                                                    | AoT S2, FSN UBW, Bleach TYBW                          |
| `PREQUEL`             | `"Prequel"`             | `PREQUEL`     | `prequel`             | `prequels`     | `"Prequel"`                  | subtype `"Prequel"`                      | `"Prequel"`                                                   | AoT S2, FSN UBW, Steins;Gate 0, DBZ                   |
| `SIDE_STORY`          | `"Side Story"`          | `SIDE_STORY`  | `side_story`          | `sideStories`  | `"Side Story"`               | subtype `"Side Story"`                   | `"Side Story"`                                                | One Piece (26), Gintama, Naruto, DBZ                  |
| `PARENT_STORY`        | `"Parent Story"`        | `PARENT`      | `parent_story`        | `parents`      | `"Parent Story"`             | subtype `"Original"`                     | `"Main Story"`                                                | OP Dead End→One Piece, Gintama: Jump Festa 2014       |
| `FULL_STORY`          | `"Full Story"`          | —             | `full_story`          | —              | `"Full Story"`               | subtype `"Full Version"`                 | `"Complete Story"`                                            | Eva Death&Rebirth→NGE, Initial D Battle Stage 3       |
| `SUMMARY`             | `"Summary"`             | `SUMMARY`     | `summary`             | —              | `"Summary"`                  | subtype `"Condensed Version"`, `"Recap"` | `"Summary"`                                                   | AoT, NGE, One Piece (32), Gintama, DBZ                |
| `SPIN_OFF`            | `"Spin-Off"`            | `SPIN_OFF`    | `spinoff`             | `spinoffs`     | —                            | subtype `"Spin-off"`                     | —                                                             | Railgun, AoT→Chuugakkou, NGE                          |
| `ALTERNATIVE_VERSION` | `"Alternative Version"` | `ALTERNATIVE` | `alternative_version` | `alternatives` | `"Alternative Version"`      | subtype `"Remake"`                       | `"Alternative Version"`, `"Remake"`                           | DB Kai, One Piece movies, DBZ, Gintama, FMA:B         |
| `ALTERNATIVE_SETTING` | `"Alternative Setting"` | —             | `alternative_setting` | —              | `"Alternative Setting"`      | subtype `"Alternate Universe"`           | `"Alternative Environment"`                                   | One Piece (3), Fate/stay night (5), FMA:B→FMA         |
| `SIDE_STORY`          | —                       | —             | —                     | —              | —                            | subtype `"Omake"`                        | —                                                             | Gintama specials                                      |
| `SAME_FRANCHISE`      | —                       | —             | —                     | —              | —                            | subtype `"Same Franchise"`, no subtype   | —                                                             | One Piece, Gintama                                    |
| `SAME_SETTING`        | —                       | —             | —                     | —              | `"Same Setting"`             | —                                        | `"Shared Universe"`, `"Same origin"`                          | One Piece→MONSTERS, Bleach, Railgun                   |
| `CHARACTER`           | `"Character"`           | `CHARACTER`   | `character`           | —              | `"Character"` _(deprecated)_ | —                                        | `"Character"`                                                 | One Piece (4), Gintama, AoT, Bleach, Railgun          |
| `CROSSOVER`           | —                       | —             | —                     | —              | —                            | —                                        | `"Crossover"`                                                 | One Piece (4), DBZ (2)                                |
| `ADAPTATION`          | `"Adaptation"`          | `ADAPTATION`  | `adaptation`          | —              | —                            | _(manga relation: "Original Manga")_     | `"Original Work"`, `"Adaptation"`, `"Adaptation: Incomplete"` | One Piece, AoT, Fate                                  |
| `OTHER`               | `"Other"`               | `OTHER`       | `other`               | `other`        | `"Other"`                    | subtype fallback (no match)              | `"Other"`, `"?"`                                              | AoT, NGE, Gintama x Mameshiba, One Piece (5)          |
| _(defensive)_         | —                       | `SOURCE`      | —                     | —              | —                            | —                                        | —                                                             | AniList schema-only, 0 data hits → `ADAPTATION`       |
| _(defensive)_         | —                       | `COMPILATION` | —                     | —              | —                            | —                                        | —                                                             | AniList schema-only, 0 data hits → `SUMMARY`          |
| _(defensive)_         | —                       | `CONTAINS`    | —                     | —              | —                            | —                                        | —                                                             | AniList schema-only, 0 data hits → `OTHER`            |

**MAL**: 12 active. Entry `type` always lowercase `"anime"` or `"manga"`.
**AniList**: 10 active + 3 schema-only (SOURCE, COMPILATION, CONTAINS — v2, never populated).
**Kitsu**: 12 roles (snake_case). Matches MAL 1:1 except `spinoff` (no hyphen). Confirmed from [server source](https://github.com/hummingbird-me/kitsu-server/blob/the-future/app/models/media_relationship.rb).
**AnimeSchedule**: 7 keys (camelCase dict keys). Values are route slugs, not IDs. Missing: FULL_STORY, SUMMARY, ALTERNATIVE_SETTING, CHARACTER, CROSSOVER, ADAPTATION. Its `other` key holds every relation outside its six other keys, so a summary or compilation film arrives as `other` (Kidou Senshi Gundam → Kidou Senshi Gundam I). Most `other` entries are relations other providers also call other or character, so AnimeSchedule's `OTHER` does not contradict a more specific type from another provider ([API docs](https://img.animeschedule.net/api/v3/documentation/anime)).
**AniDB**: 11 types per [wiki](https://wiki.anidb.net/index.php?title=Content:Relations). `"Character"` is deprecated → use `"Other"`. `"Same Setting"` (same world, different characters) maps to `SAME_SETTING`, not to `SIDE_STORY`, which AniDB defines as a story that takes place during the parent storyline. `"Alternative Version"` exists in wiki but not confirmed in live data (remakes use `"Alternative Setting"` instead). No SPIN_OFF or ADAPTATION — AniDB uses `"Side Story"` and `"Other"` respectively.
**AnimePlanet**: Two `relation_type` values: `same_franchise` and `other_franchise` — these are grouping buckets only, NOT used for mapping. The mapper always uses `relation_subtype` for semantic type. Confirmed subtypes: `Side Story`, `Condensed Version`, `Recap`, `Remake`, `Alternate Universe`, `Omake`, `Same Franchise`, `Sequel`, `Prequel`, `Other Franchise`, `Original`, `Full Version`, `Spin-off`. `Original` links a special or movie to the series it comes from (Gintama: Jump Festa 2014 → Gintama' Enchousen), so it maps to `PARENT_STORY`; `Full Version` links a short to the full series (Captain Future race special → Captain Future), so `FULL_STORY`. Counts over 492 cached pages: no subtype 1,260, Same Franchise 146, Prequel 106, Sequel 91, Other Franchise 78, Side Story 60, Recap 35, Original 30, Condensed Version 18, Remake 14, Omake 5, Alternate Universe 3, Spin-off 2, Full Version 1. Entries with `relation_subtype=None` or `"Same Franchise"` map to `SAME_FRANCHISE`: the relations tab lists the whole franchise, and most entries say no more than that they belong to it. Entries with `relation_subtype="Other Franchise"` or unrecognized subtypes default to `OTHER`. Manga relations include `"Original Manga"` subtype → `ADAPTATION`. Confirmed via One Piece (65 entries) and Gintama.
**AniSearch**: 16 anime relation types, counted over 300 cached pages. No SPIN_OFF — spin-offs are folded into `"Side Story"`. `"Main Story"` (68) is the parent and maps to `PARENT_STORY`; `"Complete Story"` (19) links a summary to the full series (Initial D Battle Stage 3 → Initial D Fifth Stage) and maps to `FULL_STORY`. Uses `"Alternative Environment"` (not `"Alternative Setting"`). `"Shared Universe"` and `"Same origin"` map to `SAME_SETTING`. `"Remake"` maps to `ALTERNATIVE_VERSION`. `"?"` = uncategorized → `OTHER`. `"Original Work"` (in manga relations section) and `"Adaptation"` / `"Adaptation: Incomplete"` (on manga page) map to `ADAPTATION`. Relation types appear as colored `<span>` labels inside `<th>` cells in the relations table.


### Source material relations

A related manga, novel or other work is filed under `related_source_material`, keyed by `SourceMaterialRelationType`.

| Canonical      | MAL              | AniList                  | AnimePlanet                  | AniSearch (manga relations)                    | Verified with                                   |
| -------------- | ---------------- | ------------------------ | ---------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| `ADAPTATION`   | `"Adaptation"`   | `ADAPTATION`             | subtype `"Original Manga"` (set by the mapper) | `"Original Work"`, `"Adaptation"`, `"Adaptation: Incomplete"` | One Piece, Log Horizon                  |
| `SOURCE`       | —                | `SOURCE` _(schema)_      | —                            | —                                              | AniList schema-only                             |
| `SPIN_OFF`     | `"Spin-Off"`     | `SPIN_OFF`               | —                            | `"Adaptation: Spin-off"`                       | 3 entries over 300 AniSearch pages              |
| `ALTERNATIVE`  | `"Alternative version"` | `ALTERNATIVE`     | —                            | —                                              | —                                               |
| `SIDE_STORY`   | —                | `SIDE_STORY`             | —                            | `"Same origin"`                                | My Happy Marriage (2026) → its manga            |
| `SEQUEL`       | —                | `SEQUEL`                 | —                            | —                                              | AniList live run, 3 entries                     |
| `PREQUEL`      | —                | `PREQUEL`                | —                            | —                                              | AniList live run                                |
| `PARENT_STORY` | —                | `PARENT`                 | —                            | —                                              | AniList live run                                |
| `CHARACTER`    | `"Character"`    | `CHARACTER`              | —                            | —                                              | AniList live run, 6 entries                     |
| `OTHER`        | `"Other"`        | `OTHER`                  | other subtypes               | `"?"`                                          | Ghost in the Shell Arise                        |

**AniSearch** manga relations over 300 cached pages: `Original Work` 161, `?` 28, `Same origin` 12, `Adaptation: Spin-off` 3, `Adaptation` 3. `Same origin` is another adaptation of the same original, e.g. the manga of the novel an anime adapts.
**AniList** files a relation to a manga under the anime relation's own type, so `SEQUEL`, `PREQUEL`, `PARENT`, `SIDE_STORY` and `CHARACTER` appear on manga edges too.

---

## Anime Type / Format

| Canonical    | MAL            | AniList    | Kitsu (`subtype`) | AnimeSchedule (`mediaTypes`)                        | AniDB           | AnimePlanet     | AniSearch       | Verified with                                      |
| ------------ | -------------- | ---------- | ----------------- | -------------------------------------------------- | --------------- | --------------- | --------------- | -------------------------------------------------- |
| `TV`         | `"TV"`         | `TV`       | `"TV"`            | `"TV"` (route: `tv`)                               | `"TV Series"`   | `"TV"`          | `"TV-Series"`   | One Piece, AoT, NGE, HxH, Kanon                    |
| `TV_SHORT`   | —              | `TV_SHORT` | —                 | `"TV Short"` (route: `tv-short`)                   | —               | —               | —               | AniList: Saiki Kusuo; AS: Isekai Quartet           |
| `TV_SPECIAL` | `"TV Special"` | —          | —                 | —                                                  | `"TV Special"`  | `"TV Special"`  | `"TV-Special"`  | SAO Extra Edition, OP 3D2Y, 86 Special             |
| `MOVIE`      | `"Movie"`      | `MOVIE`    | `"movie"`         | `"Movie"` (route: `movie`)                         | `"Movie"`       | `"Movie"`       | `"Movie"`       | Akira, Eva EoE, OP Dead End, Kimi no Na wa         |
| `OVA`        | `"OVA"`        | `OVA`      | `"OVA"`           | `"OVA"` (route: `ova`)                             | `"OVA"`         | `"OVA"`         | `"OVA"`         | Fate/kaleid, OP Defeat Ganzak, Hellsing U          |
| `ONA`        | `"ONA"`        | `ONA`      | `"ONA"`           | `"ONA"` (route: `ona`)                             | `"Web"`         | `"Web"`         | `"Web"`         | Cyberpunk Edgerunners, Quanzhi Gaoshou             |
| `SPECIAL`    | `"Special"`    | `SPECIAL`  | `"special"`       | `"Special"` (route: `special`)                     | —               | `"DVD Special"` | `"Bonus"`       | OP Straw Hat Theater, 30-sai Specials              |
| `MUSIC`      | `"Music"`      | `MUSIC`    | `"music"`         | `"Music"` (route: `music`)                         | `"Music Video"` | `"Music Video"` | `"Music Video"` | Shelter (MAL 34240, AniDB 12482)                   |
| `PV`         | `"PV"`         | —          | —                 | —                                                  | —               | —               | —               | MAL: La Rose (5415)                                |
| `CM`         | `"CM"`         | —          | —                 | —                                                  | —               | —               | `"CM"`          | MAL: Sora Iro (2616), J-COM (21281)                |
| `OTHER`      | —              | —          | —                 | —                                                  | —               | `"Other"`       | `"Other"`       | OP Kanzen Kouryakuhou, Indra Pilot Film            |
| _(unknown)_  | —              | —          | —                 | —                                                  | `"unknown"`     | —               | `"Unknown"`     | AniSearch: Evangelion new (21288), Genshin (21284) |
| _(regional)_ | —              | —          | —                 | `"ONA (Chinese)"` (route: `ona-chinese`)           | —               | —               | —               | Quanzhi Gaoshou → map to `ONA`                     |
| _(regional)_ | —              | —          | —                 | `"Movie (Chinese)"` (route: `movie-chinese`)       | —               | —               | —               | Bai She Yuan Qi → map to `MOVIE`                   |
| _(regional)_ | —              | —          | —                 | `"TV (Chinese)"` (route: `tv-chinese`)             | —               | —               | —               | Yi Ren Zhi Xia → map to `TV`                       |
| _(regional)_ | —              | —          | —                 | `"OVA (Chinese)"` (route: `ova-chinese`)           | —               | —               | —               | Feng Ji Yun Nu → map to `OVA`                      |
| _(regional)_ | —              | —          | —                 | `"Special (Chinese)"` (route: `special-chinese`)   | —               | —               | —               | Arknights 2024 Special → map to `SPECIAL`          |
| _(regional)_ | —              | —          | —                 | `"TV Short (Chinese)"` (route: `tv-short-chinese`) | —               | —               | —               | Jinxiu Shenzhou → map to `TV_SHORT`                |

**MAL**: 9 values. **AniList**: 7 anime-relevant + 3 manga formats. **Kitsu**: 6 values (TV, movie, OVA, ONA, special, music). Mixed case — `"TV"` and `"OVA"` uppercase, rest lowercase.
**AnimeSchedule**: 13 values (7 base + 6 Chinese regional variants). Title Case names with kebab-case routes. Chinese variants (`ona-chinese`, `movie-chinese`, `tv-chinese`, `ova-chinese`, `special-chinese`, `tv-short-chinese`) map to their base types. No PV, CM, or TV_SPECIAL. Full list confirmed from website HTML category navigation.
**AniList note**: `MANGA`, `NOVEL`, `ONE_SHOT` are `MediaFormat` values for manga entries — NOT anime formats. They appear in anime relation edges (related manga carry their own format). The mapper filters these out when splitting `related_anime` vs `related_source_material`. As source types, `MANGA` and `NOVEL` are already in the Source Material table (`MediaSource` enum). `ONE_SHOT` has no source equivalent — anime adapted from one-shots use `source: MANGA`.
**AniDB**: 8 types per [wiki](https://wiki.anidb.net/Anime_Type): `unknown`, `TV Series`, `OVA`, `Movie`, `Other`, `Web`, `TV Special`, `Music Video`. Uses `"TV Series"` (not just `"TV"`), `"Web"` (not `"ONA"`), `"Music Video"` (not `"Music"`). 6 confirmed via live API (TV Series, Movie, OVA, Web, TV Special, Music Video). `Other` and `unknown` not confirmed via live API but documented in wiki. No PV, CM, TV_SHORT, or SPECIAL (distinct from TV Special).
**AnimePlanet**: the type is read from the entry bar (`<span class="type">`, e.g. `"Web"` in `"Web (1 ep x 2 min)"`) for the anime and its related entries; every one of 625 cached pages has it. Over those pages: TV 177, Movie 135, Web 110, Music Video 75, OVA 51, TV Special 34, Other 34, DVD Special 9. 10 types confirmed via crawl4ai page scraping (from `<span class="type">` element and `/anime/type/{slug}` URLs): `TV`, `Movie`, `OVA`, `TV Special`, `DVD Special`, `Web`, `Music Video`, `Other`, `ONA` (URL exists but labeled "Web" on pages), `Special` (URL exists). Uses `"Web"` (not `"ONA"`) and `"Music Video"` (not `"Music"`). `"DVD Special"` maps to `SPECIAL` (disc-bundled extras). JSON-LD `@type` only distinguishes `TVSeries` vs `Movie` — not useful for fine-grained format. No PV, CM, or TV_SHORT.
**AniSearch**: 10 types confirmed via crawl4ai scraping across 20+ anime. Uses `"TV-Series"` (hyphenated, not `"TV"`), `"Web"` (not `"ONA"`), `"Music Video"` (not `"Music"`), `"TV-Special"` (hyphenated), `"Bonus"` (not `"Special"` — disc-bundled extras), `"CM"` (commercials). `"Unknown"` used for newly announced anime where format is TBD. `"Other"` for pilot films and misc. No PV, TV_SHORT, or SPIN_OFF. Type field includes episode count and duration in same string: `"TV-Series, 25 (~24 min, Total: ~10 h)"` — parser must extract type prefix before first comma. A count AniSearch has only estimated carries a tooltip before the number (`data-tooltip="Number of episodes: preliminary estimate"`, sometimes followed by `Runtime: not confirmed`); such a count reads as unknown, like `"?"`.

---

## Status

| Canonical   | MAL                  | AniList            | Kitsu          | AnimeSchedule | AniDB                            | AnimePlanet            | AniSearch     | Verified with                              |
| ----------- | -------------------- | ------------------ | -------------- | ------------ | -------------------------------- | ---------------------- | ------------- | ------------------------------------------ |
| `ONGOING`   | `"Currently Airing"` | `RELEASING`        | `"current"`    | `"Ongoing"`  | _(derived: start < now, no end)_ | _(derived from dates)_ | `"Ongoing"`   | One Piece, Omae Gotoki                     |
| `FINISHED`  | `"Finished Airing"`  | `FINISHED`         | `"finished"`   | `"Finished"` | _(derived: end date set)_        | _(derived from dates)_ | `"Completed"` | AoT, NGE, Cowboy Bebop, Steins;Gate        |
| `UPCOMING`  | `"Not yet aired"`    | `NOT_YET_RELEASED` | `"upcoming"`   | `"Upcoming"` | _(derived: start > now)_         | _(derived from dates)_ | `"Upcoming"`  | Mashle S3, Dandadan S3, Bless, Genshin     |
| `UPCOMING`  | —                    | —                  | `"tba"`        | —            | —                                | —                      | —             | Yuuri on ICE Movie, Ten Count              |
| `UPCOMING`  | —                    | —                  | `"unreleased"` | —            | —                                | —                      | —             | Youjo Senki II, Madoka Movie               |
| `ONGOING`   | —                    | —                  | —              | `"Delayed"`  | —                                | —                      | `"On Hold"`   | One Piece (on hiatus), OP on AniSearch     |
| `CANCELLED` | —                    | `CANCELLED`        | —              | —            | —                                | —                      | —             | AniList: Yuuri on ICE Movie, Tokyo BABYLON |
| `ONGOING`   | —                    | `HIATUS`           | —              | —            | —                                | —                      | —             | AniList schema-only (0 anime results)      |
| `UNKNOWN`   | —                    | —                  | —              | —            | _(derived: no dates)_            | _(derived: no dates)_  | —             | —                                          |

**MAL**: 3 values — no Cancelled/Hiatus/Unknown.
**AniList**: 4 active + HIATUS (0 anime results, manga-only in practice).
**Kitsu**: 5 values. `tba` (392 entries) = no confirmed date. `unreleased` (77) = confirmed but not yet out. `upcoming` (46) = imminent. No cancelled/hiatus.
**AnimeSchedule**: 4 values (Title Case). `Delayed` = temporarily paused (has `delayedFrom`, `delayedUntil`, `delayedTimetable` fields). Maps to `ONGOING` since show is not finished.
**AniDB**: No explicit status field. Status derived programmatically from `start_date`/`end_date` via `determine_anime_status()` in `datetime_utils.py`. Logic: start in future → UPCOMING; start in past + no end → ONGOING; end date set → FINISHED; neither → UNKNOWN.
**AnimePlanet**: No explicit status field — same as AniDB. Status derived from `start_date`/`end_date` scraped from JSON-LD `startDate`/`endDate`. Uses same `determine_anime_status()` utility. Stored value `"AIRING"` in One Piece data was computed, not scraped.
**AniSearch**: 4 values. `"On Hold"` maps to `ONGOING` — used for anime on production hiatus (e.g. One Piece on AniSearch). `"Completed"` (not `"Finished"`). No Cancelled or Unknown status values observed. AniSearch's own status is used when it maps to a known value; only a missing or unrecognised status falls back to `determine_anime_status()` on `start_date`/`end_date` (the date rule calls a finished work with no end date `ONGOING`).

---

## Dates

`aired_dates` holds only real dates: a source value becomes `aired_from`/`aired_to` only when it states the day. A month and year sets `year` and `month`; a year alone sets `year`. Nothing is made up (no 1 January for a lone year). Shared helpers in `sources/base/utils.py` read every form below: `split_date_range`, `parse_iso_date` (full dates only), `parse_partial_date` (year and month at any precision), `month_name`.

| Source       | Field read                                                                                  | Forms seen                                                                                                          | Becomes                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| MAL          | "Aired" (sidebar)                                                                           | `Oct 20, 1999 to ?`, `Oct 20, 1999 to Nov 5, 2000`, `Apr 5, 2003`, `Oct 1977`, `1988`, `2026 to ?`, `Not available` | full date → `aired_dates`; `Oct 1977` → `year` + `month`; `1988` → `year`                                 |
| MAL          | "Premiered" (sidebar, TV only)                                                              | `Fall 1999`                                                                                                         | `season` + `year`; when missing, `year` comes from "Aired"                                                |
| AniList      | `startDate`, `endDate` (`{year, month, day}`, any part may be null); `season`, `seasonYear` | `{1999, 10, 20}`, `{1977, 10, 6}`, `{2027, null, null}`                                                             | full date → `aired_dates`; year and month only → `month`; `year` from `seasonYear`, else `startDate.year` |
| Kitsu        | `startDate`, `endDate`                                                                      | `1999-10-20`, or null; `2002-01-01` for a work MAL and AniList date only as `2002`                                  | `aired_dates`; a 1 January date for a year-only work cannot be told apart from a real date, so it is kept |
| AniDB        | XML `startdate`, `enddate`                                                                  | `1999-10-20`, `1977-10`, `1970-01-01` for an upcoming work with no known date                                       | full date → `aired_dates`; `1977-10` → `year` + `season`, no date; `1970-01-01` → nothing                 |
| AnimePlanet  | JSON-LD `startDate`, `endDate`                                                              | `2020-04-05`, or absent when the page has no full date                                                              | `aired_dates`                                                                                             |
| AnimePlanet  | entry bar `span.iconYear`; season link                                                      | `2002`, `1999 - ?`; `/anime/seasons/fall-1999`                                                                      | `year` when JSON-LD has no `startDate`; `season` from the link                                            |
| AniSearch    | "Published" (information section)                                                           | `20.10.1999`, `11.2008`, `2027`, `?`, as a range: `20.10.1999 ‑ ?`, `2027 ‑ ?`, `20.10.1999‑31.03.2002`             | full date → `aired_dates` + `season`; `11.2008` → `year` + `month`; `2027` → `year`                       |
| AnimeSchedule | `premier`, `month`                                                                          | full dates; `2002-01-01` for the same year-only work as Kitsu                                                       | `aired_dates.aired_from`, `month`; 1 January dates kept, as for Kitsu                                     |

**Range separators**: MAL `" to "`; AniSearch an en dash or non-breaking hyphen (`‑`), spaced or not; AnimePlanet `" - "`.
**Measured on 2026-10-04**: 11 of 247 MAL pages gave a year alone in "Aired" and 39 a month and year; 81 of 168 AnimePlanet pages had no JSON-LD `startDate`; AniList `startDate` had a full date for 328 of 453 anime.
**`month`** is the English month name (`"October"`). In the merge, a provider that states `month` beats the month read from the merged `aired_from`, so MAL, AniList and AniSearch state it only when they have no full date; otherwise MAL would override AnimeSchedule's month.

---

## Source Material

Kitsu and AnimePlanet have **no dedicated source material field**. Kitsu confirmed from [Anime model source](https://github.com/hummingbird-me/kitsu-server/blob/the-future/app/models/anime.rb). AnimePlanet encodes source material as **genre tags** (e.g., `"Based on a Manga"`, `"Original Work"`) rather than a structured field. **AniDB** also has no dedicated field — source material is encoded as **tags** under parent tag `original work` (t2609), extracted by matching tag names from the flat tags list.

| Canonical       | MAL              | AniList                         | AnimeSchedule (`sources`) | AniDB (tag name)                                                   | AnimePlanet (genre tag)                               | AniSearch (`adapted`) | Verified with                                       |
| --------------- | ---------------- | ------------------------------- | ------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------- | --------------------- | --------------------------------------------------- |
| `MANGA`         | `"Manga"`        | `MANGA`                         | `"Manga"`                | `"manga"` (t2798)                                                  | `"Based on a Manga"`                                  | `"Manga"`             | One Piece, Bocchi the Rock, AoT, Dandadan           |
| `WEBTOON`       | —                | —                               | —                        | —                                                                  | `"Based on a Webtoon"`                                | `"Webtoon"`           | Tempal: Item no Chikara, Megami Kourin Dai 2 Ki     |
| `KOMA_4`        | `"4-koma manga"` | —                               | `"4-koma Manga"`         | —                                                                  | `"Based on a 4-koma Manga"`                           | —                     | Azumanga Daioh, Lucky Star, K-On!                   |
| `DOUJINSHI`     | `"Doujinshi"`    | `DOUJINSHI` _(schema)_          | —                        | —                                                                  | `"Based on a Doujinshi"`                              | —                     | Imaizumin (48755) — Adaptation (Doujinshi)          |
| `ONE_SHOT`      | `"One-shot"`     | `ONE_SHOT` _(manga format)_     | —                        | —                                                                  | —                                                     | —                     | One Piece: Strong World Episode 0 (8740)            |
| `MANHWA`        | `"Manhwa"`       | —                               | —                        | `"manhwa"` (t5010)                                                 | —                                                     | —                     | Solo Leveling (52299) — Adaptation (Manhwa)         |
| `MANHUA`        | `"Manhua"`       | —                               | —                        | `"manhua"` (t6493)                                                 | —                                                     | —                     | —                                                   |
| `WEB_MANGA`     | `"Web manga"`    | —                               | `"Web Manga"` (412)      | —                                                                  | —                                                     | —                     | One Punch Man (30276)                               |
| `LIGHT_NOVEL`   | `"Light novel"`  | `LIGHT_NOVEL`                   | `"Light Novel"`          | —                                                                  | `"Based on a Light Novel"`                            | `"Light Novel"`       | SAO (11757), SAO on AniSearch                       |
| `NOVEL`         | `"Novel"`        | `NOVEL` _(schema)_              | `"Novel"` (517)          | `"novel"` (t2799) — covers LN + prose                              | `"Based on a Novel"`                                  | `"Novel"` | Hyouka, Shinsekai Yori, Paprika |
| `WEB_NOVEL`     | `"Web novel"`    | `WEB_NOVEL` _(schema)_          | `"Web Novel"` (431)      | —                                                                  | `"Based on a Web Novel"`                              | —                     | Quanzhi Gaoshou                                     |
| `VISUAL_NOVEL`  | `"Visual novel"` | `VISUAL_NOVEL`                  | `"Visual Novel"` (383)   | —                                                                  | `"Based on a Visual Novel"`                           | `"Visual Novel"`      | Umineko, Steins;Gate                                |
| `GAME`          | `"Game"`         | `VIDEO_GAME`                    | `"Video Game"` (1003)    | `"game"` (t2800)                                                   | `"Based on a Video Game"`, `"Based on a Mobile Game"` | `"Video Game"`        | Pokemon, Cyberpunk, Shadowverse, Genshin            |
| `GAME`          | —                | `GAME` _(schema)_               | —                        | —                                                                  | —                                                     | —                     | AniList v3 — entries show `OTHER`                   |
| `CARD_GAME`     | `"Card game"`    | —                               | `"Card Game"` (76)       | —                                                                  | `"Based on a Card Game"`                              | —                     | Shadowverse, Manaria Friends                        |
| `ORIGINAL`      | `"Original"`     | `ORIGINAL`                      | `"Original"` (7326)      | `"new"` (t2797)                                                    | `"Original Work"`                                     | `"Original Work"`     | Cowboy Bebop, Suzume, Shelter                       |
| `MIXED_MEDIA`   | `"Mixed media"`  | `MULTIMEDIA_PROJECT` _(schema)_ | —                        | —                                                                  | —                                                     | —                     | BanG Dream (33573)                                  |
| `MUSIC`         | `"Music"`        | —                               | `"Music"` (138)          | —                                                                  | —                                                     | —                     | Heroine Tarumono, Mekakucity Actors                 |
| `RADIO`         | `"Radio"`        | —                               | —                        | `"radio programme"` (t6453)                                        | —                                                     | —                     | Suzakinishi (30826) — MAL only, 0 in AnimeSchedule   |
| `BOOK`          | `"Book"`         | —                               | `"Book"` (113)           | —                                                                  | `"Based on a Religious Text"`                         | —                     | Hi no Ame (5929)                                    |
| `FAIRY_TALE`    | —                | —                               | —                        | —                                                                  | `"Based on a Fairy Tale"`                             | `"Fairy Tale"`        | Red Shoes, Urashima Tarou, Sinbad, Toshishun        |
| `SHORT_STORY`   | —                | —                               | —                        | —                                                                  | —                                                     | `"Short Story"`       | Xiao Qian (AniSearch 3479)|
| `PICTURE_BOOK`  | `"Picture book"` | `PICTURE_BOOK` _(schema)_       | `"Picture Book"` (161)   | `"picture book"` (t7469)                                           | `"Based on a Picture Book"`                           | —                     | Anpanman (60431)                                    |
| `COMIC`         | —                | `COMIC`                         | —                        | `"western comics"` (t3430)                                         | —                                                     | —                     | AniList: western comics (DC, Marvel, etc.)          |
| `LIVE_ACTION`   | —                | `LIVE_ACTION`                   | —                        | `"live-action film"` (t2796), `"television programme"` (t6446)     | `"Based on a Movie"`, `"Based on a TV Series"`        | `"Live-Action"`       | Star Wars: Visions, Pacific Rim: The Black, Garo    |
| `ILLUSTRATION`  | —                | —                               | —                        | `"cg collection"` (t7252)                                          | —                                                     | —                     | AniDB: illustrated CG work with story/dialogue      |
| `WESTERN_MEDIA` | —                | —                               | —                        | `"western animated cartoon"` (t3714), `"american derived"` (t4424) | —                                                     | —                     | Powerpuff Girls Z, Batman Ninja, Star Wars: Visions |
| `OTHER`         | `"Other"`        | `OTHER`                         | `"Other"` (885)          | —                                                                  | `"Based on a Doujinshi"`, `"Based on a Play"`         | `"Other"`             | various, Trouble Chocolate                          |
| `UNKNOWN`       | `"Unknown"`      | —                               | —                        | —                                                                  | —                                                     | —                     | Gushu Xin Shuo (44651) — MAL only                   |
| `OTHER`         | —                | `ANIME`                         | —                        | —                                                                  | —                                                     | —                     | AniList: anime adapted from existing anime          |

**MAL**: 18 active (all verified via live API). `"Manhwa"` and `"Doujinshi"` appear as media types in Related Entries (e.g. `Adaptation (Manhwa)`), not always as the `source` sidebar field. `"One-shot"` and `"4-koma manga"` appear in both.
**AniList**: 9 active (ORIGINAL, MANGA, LIGHT_NOVEL, VISUAL_NOVEL, VIDEO_GAME, OTHER, ANIME, LIVE_ACTION, COMIC). `DOUJINSHI` and `ONE_SHOT` are in schema but stored as `OTHER` in practice; here mapped to their own canonical types as they appear in related entries. `ANIME` (anime adapted from existing anime) maps to `OTHER`. `LIVE_ACTION` and `COMIC` have their own canonical enum values.
**Kitsu**: No source field.
**AnimeSchedule**: 13 active (Title Case names, kebab-case routes). Uses `"Video Game"` (not just `"Game"`). Has `"4-koma Manga"` and `"Web Manga"` as distinct types. No `RADIO`, `MIXED_MEDIA`, `MANHWA`, `MANHUA`, or `UNKNOWN`.
**AniDB**: 14 source material tags under parent `original work` (t2609), encoded in the flat tags list (no dedicated XML field). Total tagged: ~10,039 of 16,611 entries (~40% have no source tag). `"novel"` covers both light novels and prose novels — no distinction. `"new"` = original work (1835 entries). `"american derived"` and `"western animated cartoon"` both map to `WESTERN_MEDIA` — they overlap (e.g. Powerpuff Girls Z, Rick & Morty anime appear in both). Tag names extracted as lowercase strings from XML `<tag><name>` elements.
**AnimePlanet**: Source material encoded as genre tags, not a dedicated field. The model reads the first `"Based on a ..."` / `"Original Work"` tag whose source has a canonical type, so `"Based on a 4-Koma Manga"` wins over `"Based on a Manga"` and `"Based on a Visual Novel"` over `"Based on an Otome Game"`. Counted over 625 cached pages (364 with a source tag): Manga 151, Original Work 73, Light Novel 55, Video Game 34, 4-Koma Manga 13, Novel 12, Web Novel 9, Mobile Game 6, Webtoon 6, Card Game 5, Visual Novel 5, Otome Game 4, Fairy Tale 3, Religious Text 1, Eroge 1, Cartoon 1, Picture Book 1. Otome Game and Eroge map to `VISUAL_NOVEL` (AniList files Collar x Malice, tagged Otome Game, as a visual novel), Religious Text to `BOOK`, Cartoon to `WESTERN_MEDIA`. `"Based on a Movie"` and `"Based on a TV Series"` map to `LIVE_ACTION`: the tagged works are live-action films and dramas (Star Wars, Kamen Rider, Winter Sonata). A few are anime instead ("SSSS.Gridman Movie" is based on the SSSS.Gridman anime); the tag does not tell them apart. 14 confirmed tags via crawl4ai: `Based on a Manga`, `Based on a Light Novel`, `Based on a Visual Novel`, `Based on a Video Game`, `Based on a Web Novel`, `Based on a Novel`, `Based on a Card Game`, `Based on a Doujinshi`, `Based on a 4-koma Manga`, `Based on a Webtoon`, `Based on a Picture Book`, `Based on a Mobile Game`, `Based on a Play`, `Original Work`. No Web Manga, Book, Radio, Music, or Mixed Media equivalents.
**AniSearch**: values from the "Adapted From" field (`<div class="adapted">`), counted over 300 cached pages: `Manga`, `Original Work`, `Light Novel`, `Other`, `Video Game`, `Novel`, `Web Novel`, `Fairy Tale`, `Webtoon`, `Visual Novel`. `Fairy Tale` (Red Shoes, Urashima Tarou, Sinbad) maps to `FAIRY_TALE`; `Short Story` (Xiao Qian) maps to `SHORT_STORY`. Uses `"Video Game"` (not `"Game"`, same as AnimeSchedule). No dedicated Card Game, Novel, Web Novel, Web Manga, Book, Picture Book, Music, Radio, or Mixed Media types — likely folded into `"Other"` or omitted. Some entries have no "Adapted From" field at all (e.g. Music Videos). Field absent = source unknown, NOT `"Original Work"`.

---

## Rating / Age Rating

| Canonical   | MAL                                | Kitsu (`ageRating`) | AniDB                                        | AnimePlanet                 | AniSearch                | Verified with               |
| ----------- | ---------------------------------- | ------------------- | -------------------------------------------- | --------------------------- | ------------------------ | --------------------------- |
| `G`         | `"G - All Ages"`                   | `"G"` (6,727)       | —                                            | —                           | —                        | On Your Mark (1047)         |
| `PG`        | `"PG - Children"`                  | `"PG"` (10,394)     | —                                            | —                           | —                        | Pokemon (527), One Piece    |
| `PG13`      | `"PG-13 - Teens 13 or older"`      | `"PG"` + teen guide | —                                            | —                           | —                        | One Piece (21)              |
| `R17`       | `"R - 17+ (violence & profanity)"` | `"R"` (2,497)       | —                                            | —                           | —                        | Cowboy Bebop (1), AoT       |
| `R_PLUS`    | `"R+ - Mild Nudity"`               | `"R"` + nudity guide | —                                            | —                           | —                        | MAL 19, 26, 32              |
| `RX`        | `"Rx - Hentai"`                    | `"R18"` (0 entries) | —                                            | —                           | —                        | MAL 188, 203                |
| _(omit)_    | `null`                             | `null`              | —                                            | —                           | —                        | Some upcoming/obscure anime |
| _(numeric)_ | —                                  | —                   | `permanent` / `temporary` / `review` ratings | `aggregateRating` (JSON-LD) | community rating (score) | One Piece (69)              |

**MAL**: `AnimeRating` enum uses MAL's full rating strings — no normalization needed.
**AniList**: Uses `isAdult` (boolean) instead of rating.
**Kitsu**: codes `G`, `PG`, `R` and `R18`; the public API returns no `R18` titles. Kitsu has no PG-13 or R+ code: editors note them in the free-text `ageRatingGuide`, typed in many wordings ("Teens 13 or older", "13 - Teens 13 or older", "13 and older"; "Mild Nudity", "partial nudity", "Ecchi"). `KitsuAnimeAttributes.content_rating` reads both fields:

- The code sets the level, because it comes from a fixed list.
- A guide naming teen content (`13`, `teen`) raises `PG` to `PG13`.
- A guide naming sexual content (`nudit`, `ecchi`, `sex`, `smut`, `erotica`) raises `R` to `R_PLUS`, never to `RX`, since Kitsu codes adult titles `R18`.
- A guide disagreeing with the code otherwise is ignored (code `G` with "Teens 13 or older" stays `G`).
- Without a code, the guide alone is read the same way, then through the `AnimeRating` map; only a guide that still has no meaning is logged.

Over 20,230 cached Kitsu anime: `PG` 9,645, `G` 6,080, `R` 2,355, no code 2,145 (2,129 of them with no guide either).
**AnimeSchedule**: No rating/age rating field.
**AniDB**: No age rating field. Has numeric community ratings in 3 categories: `permanent` (weighted, long-standing voters), `temporary` (recent/low-vote-count), `review` (from written reviews). Each has `value` (float, scale 1–10) and `count` (int). These map to `statistics["anidb"]` in the canonical model, not to `rating` (age classification). Example: One Piece permanent=8.33 (9547 votes), temporary=8.58 (10282), review=8.68 (20).
**AnimePlanet**: No age rating field (`contentRating` in JSON-LD is always `null`). Has community ratings via JSON-LD `aggregateRating`: `ratingValue` (float, scale 0.5–5), `ratingCount`, `reviewCount`, `bestRating: 5`, `worstRating: 0.5`. Maps to `statistics["anime_planet"]`. Example: One Piece ratingValue=4.314, ratingCount=63900, reviewCount=267.
**AniSearch**: No age rating field. Has community rating score (numeric, displayed as star rating on the page). Maps to `statistics["anisearch"]`, not to `rating` (age classification).

---

## Taxonomy Vocabularies

Where each source publishes its own list of genres / themes / tags, and what it
costs to fetch. Verified live on 2026-09-22. These lists were compared to find
the spellings in `libs/enrichment/src/enrichment/pipeline/same_word.py` that mean
the same word (`science fiction` and `Sci-Fi`, `Superpowers` and `Super Power`).
No code fetches them: `merge_categories` in `metadata_rules.py` files each word
by what the providers called it, not by a vocabulary.

| Source        | Endpoint                                                   |     Calls |                    Size | Notes                                                                                                                                                                                                 |
| :------------ | :--------------------------------------------------------- | --------: | ----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AniList       | GraphQL `{ GenreCollection }`                              |         1 |                      19 | Fixed genre list                                                                                                                                                                                      |
| AniList       | GraphQL `{ MediaTagCollection { name category isAdult } }` |         1 |                     428 | Carries the category (`Theme-*`, `Cast-*`, `Setting-*`, `Technical`, `Demographic`) that the mapper routes on, plus `isAdult` for `content_warnings`                                                  |
| AniDB         | `httpapi?request=taglist`                                  |     **1** |               **1,723** | The largest vocabulary of the seven. Carries `id`, `parentid` and `isverified`, so the tag hierarchy comes free. Not listed with the other request types — `request=tag` answers `<error code="320">` |
| Kitsu         | `api/edge/categories`                                      | paginated |                     218 | `meta.count` gives the total                                                                                                                                                                          |
| Anime-Planet  | `/anime/tags`                                              |  20 pages |                     675 | 35 per page, fixed server-side — `per_page`, `limit` and `size` are all ignored. Cloudflare rejects curl with `403`; needs a browser                                                                  |
| AniSearch     | `/anime/genre`                                             |         1 | 20 main + 26 subsidiary | Plain HTTP. `main` and `subsidiary` are separate link classes on one page. No tag index found                                                                                                         |
| AnimeSchedule | `/genres`                                                  |         1 |                      30 | Plain HTTP page; the v3 API has no `/genres` endpoint (`404`)                                                                                                                                         |

**Prefer the API over the browser.** AniDB's `taglist` replaces roughly 97 browser
page-fetches at 30 tags each, and needs no Cloudflare handling at all. It was
found by probing request types after the documented ones came back invalid —
worth repeating for other sources before writing a crawler.

**Reaching AniDB pages with a browser**, where no API exists: it must be
`headless=False`. A headless browser gets the antileech block page (~28 KB);
the same navigation with a head returns the real page (~50 KB). Follow the
pattern in `anidb_character_crawler.py` — poll until the page settles into
either content or a recognised block rather than sleeping a fixed time, then
solve and wait for the reload. `zendriver.core.cloudflare.verify_cf` plus the
"Please Unban Me" button covers both block types.

**Cloudflare status**: AniDB and Anime-Planet block plain HTTP entirely.
AniSearch and AnimeSchedule serve these index pages to curl with a normal
user-agent. MAL, AniList and Kitsu are APIs and need no browser.

---

## Company Fields

Where each source publishes studios, producers and licensors, and the URL it
gives for a company. Verified live on 2026-09-24 over 684 anime.

| Source        | Where                                                      | Roles                      | Company URL                                  |
| :------------ | :--------------------------------------------------------- | :------------------------- | :------------------------------------------- |
| MAL           | anime page                                                 | studio, producer, licensor | `myanimelist.net/anime/producer/{id}/{Name}` |
| AniList       | GraphQL `studios { edges { node { isAnimationStudio } } }` | studio, producer           | `anilist.co/studio/{id}`                     |
| Kitsu         | `api/edge/anime/{id}/anime-productions?include=producer`   | studio, producer, licensor | `api/edge/producers/{id}`                    |
| AniDB         | `httpapi?request=anime`, the `<creators>` block            | studio, producer           | `anidb.net/creator/{id}`                     |
| Anime-Planet  | anime page, `entryBar` studio anchors                      | studio                     | `anime-planet.com/anime/studios/{slug}`      |
| AniSearch     | anime page, single studio field                            | studio                     | `anisearch.com/company/{id},{slug}`          |
| AnimeSchedule | `api/v3/anime/{route}`                                     | studio                     | `animeschedule.net/studios/{route}`          |

Coverage over the same 684 anime:

| Source        | studios | producers | licensors | anime covered |
| :------------ | ------: | --------: | --------: | ------------: |
| MAL           |     659 |     1,100 |       276 |           665 |
| AniList       |     628 |       933 |         0 |           611 |
| Anime-Planet  |     624 |         0 |         0 |           578 |
| AniSearch     |     560 |         0 |         0 |           560 |
| AnimeSchedule |     556 |         0 |         0 |           493 |
| Kitsu         |     356 |     1,222 |       140 |           495 |
| AniDB         |     168 |       155 |         0 |           230 |

Per-source notes, each of which cost something to find:

- **Only MAL and Kitsu carry all three roles.** Every licensor we publish rests
  on those two.
- **AniDB's `<creators>` mixes companies and people**, told apart only by the
  `type` attribute. `Animation Work` is the studio and `Work` the producer; over
  219 cached responses those are 72 and 66 distinct names and all but one are
  companies. Every other type is people - including two that read like company
  fields. `Animation Production` held only `Shinkai Makoto`, and `Original Plan`
  mixes `Bandai` and `Bushiroad` with `Tezuka Osamu` and `Jules Verne`. Neither
  is mapped.
- **Kitsu carries the role on the join row, not the company**, so one company can
  appear twice for one anime under different roles - it files Madhouse as both
  producer and studio on Death Note. It also repeats join rows verbatim, and
  holds separate records for one company: `MADHOUSE` is producer 5 and
  `Madhouse` is producer 917, the second with an auto-suffixed slug because the
  name already existed.
- **Kitsu has no public producer page.** `kitsu.app/producers/{slug}` renders an
  empty title and site chrome only, while an anime page renders properly, so the
  `403` from curl was real rather than Cloudflare. The API resource is the only
  stable URL.
- **AniSearch has a single studio field** and structurally cannot report a second
  company: 560 names across 560 anime.
- **AniDB's figure is a floor.** It was banned throughout collection and served
  only what the HTTP cache already held, while carrying an AniDB URL for 579 of
  660 titles.
- **AniList's `isMain` is not a role signal.** Its schema defines
  `isAnimationStudio` as whether the company is an animation studio - the
  studio-versus-producer question - and `isMain` as which studio led this anime.
  On One Piece `isMain` is true for Toei Animation alone while TAP, Magic Bus,
  Mushi Production, Studio Guts and Asahi Production are all
  `isAnimationStudio=True, isMain=False`.

How these are reconciled into one company list is a separate question - see
[merge_rules.md](merge_rules.md).

---

## AniDB IDs

AniDB XML carries numeric `type` or `id` attributes whose meaning no other provider shares. These tables say what each value means.

### External Resource Types (Anime-Level)

Found in the `<resources>` tag for an entire anime series. The `identifier` is the unique key used on the external service.

The `Evidence` column records how each row was established. **verified** is the
strongest: the type's identifier, taken from live XML, was matched against the
link AniDB renders on that same anime's page, with the anchor's own
`i_resource_*` class naming the service. 102 anime were scanned on 2026-09-17,
the pool deliberately mixed (Chinese, Korean, visual-novel adaptations, old
Japanese, recent) because a random sample only ever turns up the common types.

Other values:

- **observed** — the link was read off AniDB's own rendered anime page, where each
  anchor carries a named class (`i_resource_allcinema`, `i_resource_syoboi`, …),
  so service and URL come from the same element. Sample: 31 anime, 2026-09-16.
- **third-party** — taken from [animetosho](https://github.com/animetosho/animetosho-website)
  (`pages/series.php`), which ingests AniDB data and carries a complete type
  table. Cross-checked against [ShokoServer](https://github.com/ShokoAnime/ShokoServer)
  (`AniDBEnums.cs`), which is partial.
- **both** — independently agreed by the two above.

The join reports ambiguity rather than resolving it, which matters: type 44's
second identifier is `tv`, and that also occurs inside the official site's
`/tv/onep/` path, so a plain substring match picks the wrong anchor. Type 14 has
the same hazard — its second identifier is the single letter `v`.

| Type ID | Domain / Service      | Example URL Structure                                       | Evidence |
|:--------|:----------------------|:------------------------------------------------------------|:---------|
| 1       | Anime News Network    | `https://www.animenewsnetwork.com/encyclopedia/anime.php?id={id}` | verified |
| 2       | MyAnimeList           | `https://myanimelist.net/anime/{id}`                        | verified |
| 3       | AnimeNfo              | `https://www.animenfo.com/animetitle,{id},a.html`           | ❌ OFFLINE — site dead, not mapped |
| 4       | Official Website (JP) | Full URL supplied directly, not an identifier               | verified |
| 5       | Official Website (EN) | Full URL supplied directly, not an identifier               | verified |
| 6       | Wikipedia (English)   | `https://en.wikipedia.org/wiki/{identifier}`                | verified |
| 7       | Wikipedia (Japanese)  | `https://ja.wikipedia.org/wiki/{identifier}`                | verified |
| 8       | Syoboi Calendar       | `https://cal.syoboi.jp/tid/{id}/time`                       | verified |
| 9       | allcinema.net         | `https://www.allcinema.net/cinema/{id}`                     | verified |
| 10      | Anison.info           | `http://anison.info/data/program/{id}.html`                 | both |
| 11      | .lain                 | `http://lain.gr.jp/{identifier}` (e.g. `mediadb/media/4181`) | verified |
| 12      | Generasia             | No URL form known — AniDB lists the type but renders no link | third-party, unmapped — unseen in 102 anime |
| 13      | VGMdb                 | No URL form known — AniDB lists the type but renders no link | third-party, unmapped — unseen in 102 anime |
| 14      | VNDB                  | `https://vndb.org/{identifier}` (identifier includes the `v`, e.g. `v8983`) | verified |
| 15      | Marumegane            | `http://www.anime.marumegane.com/{id}.html`                 | ❌ DEAD — domain parked, not mapped |
| 16      | Animemorial           | `http://www.animemorial.net/ja/{id}-a`                      | verified |
| 17      | TV Animation Museum   | `http://home-aki.la.coocan.jp/anime-list/{identifier}.htm`  | third-party — unseen in 102 anime |
| 18      | TV Tropes             | No URL form known — AniDB lists the type but renders no link | third-party, unmapped — unseen in 102 anime |
| 19      | Wikipedia (Korean)    | `https://ko.wikipedia.org/wiki/{identifier}`                | verified |
| 20      | Wikipedia (Chinese)   | `https://zh.wikipedia.org/wiki/{identifier}`                | URL checked — resolves, page names the work |
| 22      | Facebook              | `https://www.facebook.com/{identifier}`                     | third-party — unseen in 102 anime |
| 23      | Twitter               | `https://twitter.com/{handle}`                              | verified |
| 26      | YouTube               | `https://www.youtube.com/{identifier}` — identifier can carry the full path (e.g. `watch?v=...`) | verified |
| 28      | Crunchyroll           | Anime-level: `https://www.crunchyroll.com/series/{id}` — Episode-level: `https://www.crunchyroll.com/watch/{id}` | verified |
| 31      | Media Arts Database   | `https://mediaarts-db.bunka.go.jp/an/anime_series/{id}` — Japanese government database | ⚠ service identified, URL broken — not mapped |
| 32      | Amazon Video          | `https://www.amazon.com/dp/{asin}`                          | verified |
| 33      | Baidu Baike           | `https://baike.baidu.com/item/{identifier}` — identifier may contain `?fromModule=...` query junk | verified |
| 34      | Official Stream       | Full URL supplied directly — any platform, not one specific service | verified |
| 35      | Official Blog         | Full URL supplied directly                                  | verified |
| 38      | Bangumi (bgm.tv)      | `https://bgm.tv/subject/{id}`                               | verified |
| 39      | Douban Movie          | `https://movie.douban.com/subject/{id}`                     | verified |
| 41      | Netflix               | `https://www.netflix.com/title/{id}`                        | verified |
| 42      | HiDive                | `https://www.hidive.com/{identifier}` (e.g. `movies/a-little-snow-fairy-sugar-ova`) | verified |
| 43      | IMDb                  | `https://www.imdb.com/title/{id}`                           | verified |
| 44      | TheMovieDB.org (TMDB) | `https://www.themoviedb.org/{type}/{id}` — two identifiers: numeric id + media type (`tv` or `movie`) | verified |
| 45      | Funimation            | `https://www.funimation.com/shows/{slug}` — platform shut down, but AniDB still stores and renders these | verified |
| 46      | QQ Video              | `https://v.qq.com/detail/{identifier}`                      | verified |
| 47      | Bilibili (Chinese)    | `https://www.bilibili.com/{identifier}` (e.g. `bangumi/media/md21082961`) | verified |
| 48      | Amazon Prime Video    | `https://www.primevideo.com/detail/{id}`                    | third-party — unseen in 102 anime |

Types 21, 24, 25, 27, 29, 30, 36, 37 and 40 are unused.

#### Corrections

Five rows were wrong. Every one of them had been marked "✅ verified", which is
why the errors survived unnoticed:

| Type | Was | Now | How it was caught |
|:-----|:----|:----|:------------------|
| 45   | Hulu | **Funimation** | XML identifier `one-piece/` renders as `funimation.com/shows/one-piece/` on AniDB's own page. Hulu appeared zero times in 133 anime |
| 31   | Funimation (deprecated) | **Media Arts Database** | Funimation is type 45, so 31 was never Funimation |
| 34   | Tencent Streaming | **Official Stream** (any platform) | AniDB's own anchor class is `official_stream`; values seen include WeTV and others |
| 9    | `prog/show_c.php?num_c={id}` | `/cinema/{id}` | Site migrated; identifier `162790` renders as `allcinema.net/cinema/162790` |
| 8    | `http://cal.syoboi.jp/tid/{id}` | `https://.../tid/{id}/time` | Identifier `350` renders as `cal.syoboi.jp/tid/350/time` |

ShokoServer's enum claims `34 = Funimation`, which contradicts both animetosho
and direct observation. Its list is partial and appears stale; prefer the
evidence above.

#### Types deliberately left unmapped

The mapper emits nothing for these rather than guessing a url. A wrong template
is worse than a missing one: type 45's Hulu entry turned Funimation identifiers
into `hulu.com` addresses that never existed.

| Type | Reason |
|:-----|:-------|
| 3 AnimeNfo | Site offline |
| 15 Marumegane | Domain is parked — serves a redirect to `/lander`, no content |
| 31 Media Arts Database | Service identified, but the database moved from `mediaarts-db.bunka.go.jp` to `mediaarts-db.artmuseums.go.jp` and the stored ids no longer resolve: the old host redirects to its root, and the new host renders an empty record for ids 12519, 2011 and 7708 |
| 12 Generasia, 13 VGMdb, 18 TV Tropes | AniDB records the type but renders no link, and animetosho has no url form either |

#### Still unconfirmed

Types 17, 22 and 48 are mapped from animetosho's table but were not seen in the
133 anime sampled, so their templates remain unverified. They stay in the mapper
because nothing suggests they are wrong — unlike 15 and 31, which were tested and
failed.


### Character Types

Found in the `<character>` tag's `charactertype` attribute (or `character_type_id` in parsed JSON).

| Type ID | Category     | Description                                               |
|:--------|:-------------|:----------------------------------------------------------|
| 1       | Character    | Standard individual personas (e.g., Luffy, Zoro).         |
| 2       | Mecha        | Robots or mechanical entities.                            |
| 3       | Organization | Groups, crews, or collectives (e.g., Straw Hat Pirates).  |
| 4       | Vessel       | Ships or primary transportation (e.g., Going Merry).      |


### Episode Number (`<epno>`) Types

Found in the `<episode>` tag's `<epno>` element `type` attribute.

| Type ID | Category            |
|:--------|:--------------------|
| 1       | Regular Episode     |
| 2       | Special             |
| 3       | Credit (OP/ED)      |
| 4       | Trailer             |
| 5       | Parody / Other      |

---
