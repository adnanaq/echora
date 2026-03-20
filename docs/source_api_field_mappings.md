# Source API Field Mappings — Verified Values

**Last updated**: 2026-03-08
**Method**: Live API calls + schema introspection against diverse anime

| Source       | Verification method                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| MAL/Jikan    | Jikan API v4 calls (One Piece, AoT, Fate, Bleach, DB Kai, etc.)                                               |
| AniList      | GraphQL `__type` introspection + live queries (One Piece, AoT, Fate, Railgun, NGE, etc.)                      |
| Kitsu        | REST API calls + [server source](https://github.com/hummingbird-me/kitsu-server) (One Piece, AoT, Fate, etc.) |
| AnimSchedule | REST API v3 calls + filter queries (One Piece, AoT, Fate/kaleid, Cyberpunk, Isekai Quartet, etc.)             |
| AniDB        | HTTP XML API calls + [wiki](https://wiki.anidb.net/Anime_Type) (One Piece, AoT, NGE, Bleach, FMA:B, etc.)     |
| AnimePlanet  | crawl4ai scraping + stored data (One Piece, AoT, Gintama, Cowboy Bebop, Shelter, Cyberpunk, etc.)             |
| AniSearch    | crawl4ai scraping (One Piece, Bleach, Naruto, DBZ, Gintama, AoT, FMA:B, Cowboy Bebop, Steins;Gate, etc.)      |

---

## Relation Types

| Canonical             | MAL/Jikan               | AniList       | Kitsu                 | AnimSchedule   | AniDB                        | AnimePlanet                              | AniSearch                                                     | Verified with                                         |
| --------------------- | ----------------------- | ------------- | --------------------- | -------------- | ---------------------------- | ---------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| `SEQUEL`              | `"Sequel"`              | `SEQUEL`      | `sequel`              | `sequels`      | `"Sequel"`                   | subtype `"Sequel"`                       | `"Sequel"`                                                    | AoT S2, FSN UBW, Bleach TYBW                          |
| `PREQUEL`             | `"Prequel"`             | `PREQUEL`     | `prequel`             | `prequels`     | `"Prequel"`                  | —                                        | `"Prequel"`                                                   | AoT S2, FSN UBW, Steins;Gate 0, DBZ                   |
| `SIDE_STORY`          | `"Side Story"`          | `SIDE_STORY`  | `side_story`          | `sideStories`  | `"Side Story"`               | subtype `"Side Story"`                   | `"Side Story"`                                                | One Piece (26), Gintama, Naruto, DBZ                  |
| `PARENT_STORY`        | `"Parent Story"`        | `PARENT`      | `parent_story`        | `parents`      | `"Parent Story"`             | —                                        | —                                                             | OP Dead End→One Piece                                 |
| `FULL_STORY`          | `"Full Story"`          | —             | `full_story`          | —              | `"Full Story"`               | —                                        | —                                                             | Eva Death&Rebirth→NGE                                 |
| `SUMMARY`             | `"Summary"`             | `SUMMARY`     | `summary`             | —              | `"Summary"`                  | subtype `"Condensed Version"`, `"Recap"` | `"Summary"`                                                   | AoT, NGE, One Piece (32), Gintama, DBZ                |
| `SPIN_OFF`            | `"Spin-Off"`            | `SPIN_OFF`    | `spinoff`             | `spinoffs`     | —                            | —                                        | —                                                             | Railgun, AoT→Chuugakkou, NGE                          |
| `ALTERNATIVE_VERSION` | `"Alternative Version"` | `ALTERNATIVE` | `alternative_version` | `alternatives` | `"Alternative Version"`      | subtype `"Remake"`                       | `"Alternative Version"`, `"Remake"`                           | DB Kai, One Piece movies, DBZ, Gintama, FMA:B         |
| `ALTERNATIVE_SETTING` | `"Alternative Setting"` | —             | `alternative_setting` | —              | `"Alternative Setting"`      | subtype `"Alternate Universe"`           | `"Alternative Environment"`                                   | One Piece (3), Fate/stay night (5), FMA:B→FMA         |
| `SIDE_STORY`          | —                       | —             | —                     | —              | `"Same Setting"`             | subtype `"Omake"`, `"Same Franchise"`    | `"Shared Universe"`, `"Same origin"`                          | One Piece→MONSTERS, Gintama specials, Bleach, Railgun |
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
**AnimSchedule**: 7 keys (camelCase dict keys). Values are route slugs, not IDs. Missing: FULL_STORY, SUMMARY, ALTERNATIVE_SETTING, CHARACTER, CROSSOVER, ADAPTATION.
**AniDB**: 11 types per [wiki](https://wiki.anidb.net/index.php?title=Content:Relations). `"Character"` is deprecated → use `"Other"`. `"Same Setting"` (same world, different characters) maps to `SIDE_STORY`. `"Alternative Version"` exists in wiki but not confirmed in live data (remakes use `"Alternative Setting"` instead). No SPIN_OFF or ADAPTATION — AniDB uses `"Side Story"` and `"Other"` respectively.
**AnimePlanet**: Two `relation_type` values: `same_franchise` and `other_franchise` — these are grouping buckets only, NOT used for mapping. The mapper always uses `relation_subtype` for semantic type. Confirmed subtypes: `Side Story`, `Condensed Version`, `Recap`, `Remake`, `Alternate Universe`, `Omake`, `Same Franchise`, `Sequel`, `Other Franchise`. Entries with `relation_subtype=None` or `"Same Franchise"` default to `SIDE_STORY`. Entries with `relation_subtype="Other Franchise"` or unrecognized subtypes default to `OTHER`. Manga relations include `"Original Manga"` subtype → `ADAPTATION`. Confirmed via One Piece (65 entries) and Gintama.
**AniSearch**: 14 relation types confirmed via crawl4ai scraping across 10+ anime (One Piece, Bleach, Naruto, DBZ, Gintama, AoT, FMA:B, Railgun, Fate/stay night, etc.). No SPIN_OFF — spin-offs are folded into `"Side Story"`. No PARENT_STORY or FULL_STORY. Uses `"Alternative Environment"` (not `"Alternative Setting"`). `"Shared Universe"` and `"Same origin"` map to `SIDE_STORY`. `"Remake"` maps to `ALTERNATIVE_VERSION`. `"?"` = uncategorized → `OTHER`. `"Original Work"` (in manga relations section) and `"Adaptation"` / `"Adaptation: Incomplete"` (on manga page) map to `ADAPTATION`. Relation types appear as colored `<span>` labels inside `<th>` cells in the relations table.

---

## Anime Type / Format

| Canonical    | MAL/Jikan      | AniList    | Kitsu (`subtype`) | AnimSchedule (`mediaTypes`)                        | AniDB           | AnimePlanet     | AniSearch       | Verified with                                      |
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
**AnimSchedule**: 13 values (7 base + 6 Chinese regional variants). Title Case names with kebab-case routes. Chinese variants (`ona-chinese`, `movie-chinese`, `tv-chinese`, `ova-chinese`, `special-chinese`, `tv-short-chinese`) map to their base types. No PV, CM, or TV_SPECIAL. Full list confirmed from website HTML category navigation.
**AniList note**: `MANGA`, `NOVEL`, `ONE_SHOT` are `MediaFormat` values for manga entries — NOT anime formats. They appear in anime relation edges (related manga carry their own format). The mapper filters these out when splitting `related_anime` vs `related_source_material`. As source types, `MANGA` and `NOVEL` are already in the Source Material table (`MediaSource` enum). `ONE_SHOT` has no source equivalent — anime adapted from one-shots use `source: MANGA`.
**AniDB**: 8 types per [wiki](https://wiki.anidb.net/Anime_Type): `unknown`, `TV Series`, `OVA`, `Movie`, `Other`, `Web`, `TV Special`, `Music Video`. Uses `"TV Series"` (not just `"TV"`), `"Web"` (not `"ONA"`), `"Music Video"` (not `"Music"`). 6 confirmed via live API (TV Series, Movie, OVA, Web, TV Special, Music Video). `Other` and `unknown` not confirmed via live API but documented in wiki. No PV, CM, TV_SHORT, or SPECIAL (distinct from TV Special).
**AnimePlanet**: 10 types confirmed via crawl4ai page scraping (from `<span class="type">` element and `/anime/type/{slug}` URLs): `TV`, `Movie`, `OVA`, `TV Special`, `DVD Special`, `Web`, `Music Video`, `Other`, `ONA` (URL exists but labeled "Web" on pages), `Special` (URL exists). Uses `"Web"` (not `"ONA"`) and `"Music Video"` (not `"Music"`). `"DVD Special"` maps to `SPECIAL` (disc-bundled extras). JSON-LD `@type` only distinguishes `TVSeries` vs `Movie` — not useful for fine-grained format. No PV, CM, or TV_SHORT.
**AniSearch**: 10 types confirmed via crawl4ai scraping across 20+ anime. Uses `"TV-Series"` (hyphenated, not `"TV"`), `"Web"` (not `"ONA"`), `"Music Video"` (not `"Music"`), `"TV-Special"` (hyphenated), `"Bonus"` (not `"Special"` — disc-bundled extras), `"CM"` (commercials). `"Unknown"` used for newly announced anime where format is TBD. `"Other"` for pilot films and misc. No PV, TV_SHORT, or SPIN_OFF. Type field includes episode count and duration in same string: `"TV-Series, 25 (~24 min, Total: ~10 h)"` — parser must extract type prefix before first comma.

---

## Status

| Canonical  | MAL/Jikan            | AniList            | Kitsu          | AnimSchedule | AniDB                            | AnimePlanet            | AniSearch     | Verified with                              |
| ---------- | -------------------- | ------------------ | -------------- | ------------ | -------------------------------- | ---------------------- | ------------- | ------------------------------------------ |
| `ONGOING`  | `"Currently Airing"` | `RELEASING`        | `"current"`    | `"Ongoing"`  | _(derived: start < now, no end)_ | _(derived from dates)_ | `"Ongoing"`   | One Piece, Omae Gotoki                     |
| `FINISHED` | `"Finished Airing"`  | `FINISHED`         | `"finished"`   | `"Finished"` | _(derived: end date set)_        | _(derived from dates)_ | `"Completed"` | AoT, NGE, Cowboy Bebop, Steins;Gate        |
| `UPCOMING` | `"Not yet aired"`    | `NOT_YET_RELEASED` | `"upcoming"`   | `"Upcoming"` | _(derived: start > now)_         | _(derived from dates)_ | `"Upcoming"`  | Mashle S3, Dandadan S3, Bless, Genshin     |
| `UPCOMING` | —                    | —                  | `"tba"`        | —            | —                                | —                      | —             | Yuuri on ICE Movie, Ten Count              |
| `UPCOMING`  | —                    | —                  | `"unreleased"` | —            | —                        | —                        | —                | Youjo Senki II, Madoka Movie                        |
| `ONGOING`   | —                    | —                  | —              | `"Delayed"`  | —                        | —                        | `"On Hold"`      | One Piece (on hiatus), OP on AniSearch              |
| `CANCELLED` | —                    | `CANCELLED`        | —              | —            | —                        | —                        | —                | AniList: Yuuri on ICE Movie, Tokyo BABYLON          |
| `ONGOING`   | —                    | `HIATUS`           | —              | —            | —                        | —                        | —                | AniList schema-only (0 anime results)               |
| `UNKNOWN`  | —                    | —                  | —              | —            | _(derived: no dates)_            | _(derived: no dates)_  | —             | —                                          |

**MAL**: 3 values — no Cancelled/Hiatus/Unknown.
**AniList**: 4 active + HIATUS (0 anime results, manga-only in practice).
**Kitsu**: 5 values. `tba` (392 entries) = no confirmed date. `unreleased` (77) = confirmed but not yet out. `upcoming` (46) = imminent. No cancelled/hiatus.
**AnimSchedule**: 4 values (Title Case). `Delayed` = temporarily paused (has `delayedFrom`, `delayedUntil`, `delayedTimetable` fields). Maps to `ONGOING` since show is not finished.
**AniDB**: No explicit status field. Status derived programmatically from `start_date`/`end_date` via `determine_anime_status()` in `datetime_utils.py`. Logic: start in future → UPCOMING; start in past + no end → ONGOING; end date set → FINISHED; neither → UNKNOWN.
**AnimePlanet**: No explicit status field — same as AniDB. Status derived from `start_date`/`end_date` scraped from JSON-LD `startDate` and page date display. Uses same `determine_anime_status()` utility. Stored value `"AIRING"` in One Piece data was computed, not scraped.
**AniSearch**: 4 values. `"On Hold"` maps to `ONGOING` — used for anime on production hiatus (e.g. One Piece on AniSearch). `"Completed"` (not `"Finished"`). No Cancelled or Unknown status values observed.

---

## Source Material

Kitsu, AniDB, and AnimePlanet have **no dedicated source material field**. Kitsu confirmed from [Anime model source](https://github.com/hummingbird-me/kitsu-server/blob/the-future/app/models/anime.rb). AniDB confirmed via XML API response — no `source` element. AnimePlanet encodes source material as **genre tags** (e.g., `"Based on a Manga"`, `"Original Work"`) rather than a structured field.

| Canonical      | MAL/Jikan                   | AniList                         | AnimSchedule (`sources`)    | AnimePlanet (genre tag)                                                   | AniSearch (`adapted`) | Verified with                                     |
| -------------- | --------------------------- | ------------------------------- | --------------------------- | ------------------------------------------------------------------------- | --------------------- | ------------------------------------------------- |
| `MANGA`        | `"Manga"`                   | `MANGA`                         | `"Manga"`                   | `"Based on a Manga"`, `"Based on a Webtoon"`                              | `"Manga"`             | One Piece, Bocchi the Rock, AoT, Dandadan         |
| `KOMA_4`       | `"4-koma manga"`            | —                               | `"4-koma Manga"`            | `"Based on a 4-koma Manga"`                                               | —                     | Azumanga Daioh, Lucky Star, K-On!                 |
| `DOUJINSHI`    | `"Doujinshi"`               | `DOUJINSHI` _(schema)_          | —                           | `"Based on a Doujinshi"`                                                  | —                     | Imaizumin (48755) — Adaptation (Doujinshi)        |
| `ONE_SHOT`     | `"One-shot"`                | `ONE_SHOT` _(manga format)_     | —                           | —                                                                         | —                     | One Piece: Strong World Episode 0 (8740)          |
| `MANHWA`       | `"Manhwa"`                  | —                               | —                           | —                                                                         | —                     | Solo Leveling (52299) — Adaptation (Manhwa)       |
| `MANHUA`       | `"Manhua"`                  | —                               | —                           | —                                                                         | —                     | —                                                 |
| `WEB_MANGA`    | `"Web manga"`               | —                               | `"Web Manga"` (412)         | —                                                                         | —                     | One Punch Man (30276)                             |
| `LIGHT_NOVEL`  | `"Light novel"`             | `LIGHT_NOVEL`                   | `"Light Novel"`             | `"Based on a Light Novel"`                                                | `"Light Novel"`       | SAO (11757), SAO on AniSearch                     |
| `NOVEL`        | `"Novel"`                   | `NOVEL` _(schema)_              | `"Novel"` (517)             | `"Based on a Novel"`                                                      | —                     | Hyouka, Shinsekai Yori, Paprika                   |
| `WEB_NOVEL`    | `"Web novel"`               | `WEB_NOVEL` _(schema)_          | `"Web Novel"` (431)         | `"Based on a Web Novel"`                                                  | —                     | Quanzhi Gaoshou                                   |
| `VISUAL_NOVEL` | `"Visual novel"`            | `VISUAL_NOVEL`                  | `"Visual Novel"` (383)      | `"Based on a Visual Novel"`                                               | `"Visual Novel"`      | Umineko, Steins;Gate                              |
| `GAME`         | `"Game"`                    | `VIDEO_GAME`                    | `"Video Game"` (1003)       | `"Based on a Video Game"`, `"Based on a Mobile Game"`                     | `"Video Game"`        | Pokemon, Cyberpunk, Shadowverse, Genshin          |
| `GAME`         | —                           | `GAME` _(schema)_               | —                           | —                                                                         | —                     | AniList v3 — entries show `OTHER`                 |
| `CARD_GAME`    | `"Card game"`               | —                               | `"Card Game"` (76)          | `"Based on a Card Game"`                                                  | —                     | Shadowverse, Manaria Friends                      |
| `ORIGINAL`     | `"Original"`                | `ORIGINAL`                      | `"Original"` (7326)         | `"Original Work"`                                                         | `"Original Work"`     | Cowboy Bebop, Suzume, Shelter                     |
| `MIXED_MEDIA`  | `"Mixed media"`             | `MULTIMEDIA_PROJECT` _(schema)_ | —                           | —                                                                         | —                     | BanG Dream (33573)                                |
| `MUSIC`        | `"Music"`                   | —                               | `"Music"` (138)             | —                                                                         | —                     | Heroine Tarumono, Mekakucity Actors               |
| `RADIO`        | `"Radio"`                   | —                               | —                           | —                                                                         | —                     | Suzakinishi (30826) — MAL only, 0 in AnimSchedule |
| `BOOK`         | `"Book"`                    | —                               | `"Book"` (113)              | —                                                                         | —                     | Hi no Ame (5929)                                  |
| `PICTURE_BOOK` | `"Picture book"`            | `PICTURE_BOOK` _(schema)_       | `"Picture Book"` (161)      | `"Based on a Picture Book"`                                               | —                     | Anpanman (60431)                                  |
| `OTHER`        | `"Other"`                   | `OTHER`                         | `"Other"` (885)             | `"Based on a Doujinshi"`, `"Based on a Play"`                             | `"Other"`             | various, Trouble Chocolate                        |
| `UNKNOWN`      | `"Unknown"`                 | —                               | —                           | —                                                                         | —                     | Gushu Xin Shuo (44651) — MAL only                 |
| `OTHER`        | —                           | `ANIME` _(schema)_              | —                           | —                                                                         | —                     | AniList entries show `OTHER`                      |
| `OTHER`        | —                           | `LIVE_ACTION` _(schema)_        | —                           | —                                                                         | —                     | AniList entries show `OTHER`                      |
| `OTHER`        | —                           | `COMIC` _(schema)_              | —                           | —                                                                         | —                     | AniList entries show `OTHER`                      |

**MAL**: 18 active (all verified via live API). `"Manhwa"` and `"Doujinshi"` appear as media types in Related Entries (e.g. `Adaptation (Manhwa)`), not always as the `source` sidebar field. `"One-shot"` and `"4-koma manga"` appear in both.
**AniList**: 6 active (ORIGINAL, MANGA, LIGHT_NOVEL, VISUAL_NOVEL, VIDEO_GAME, OTHER). `DOUJINSHI` and `ONE_SHOT` are in schema but stored as `OTHER` in practice; here mapped to their own canonical types as they appear in related entries.
**Kitsu**: No source field.
**AnimSchedule**: 13 active (Title Case names, kebab-case routes). Uses `"Video Game"` (not just `"Game"`). Has `"4-koma Manga"` and `"Web Manga"` as distinct types. No `RADIO`, `MIXED_MEDIA`, `MANHWA`, `MANHUA`, or `UNKNOWN`.
**AnimePlanet**: Source material encoded as genre tags, not a dedicated field. 14 confirmed tags via crawl4ai: `Based on a Manga`, `Based on a Light Novel`, `Based on a Visual Novel`, `Based on a Video Game`, `Based on a Web Novel`, `Based on a Novel`, `Based on a Card Game`, `Based on a Doujinshi`, `Based on a 4-koma Manga`, `Based on a Webtoon`, `Based on a Picture Book`, `Based on a Mobile Game`, `Based on a Play`, `Original Work`. No Web Manga, Book, Radio, Music, or Mixed Media equivalents.
**AniSearch**: 7 values confirmed via "Adapted From" field (`<div class="adapted">`): `Manga`, `Light Novel`, `Visual Novel`, `Video Game`, `Original Work`, `Other`. Uses `"Video Game"` (not `"Game"`, same as AnimSchedule). No dedicated Card Game, Novel, Web Novel, Web Manga, Book, Picture Book, Music, Radio, or Mixed Media types — likely folded into `"Other"` or omitted. Some entries have no "Adapted From" field at all (e.g. Music Videos). Field absent = source unknown, NOT `"Original Work"`.

---

## Rating / Age Rating

| Canonical   | MAL/Jikan                          | Kitsu (`ageRating`) | AniDB                                        | AnimePlanet                 | AniSearch                | Verified with               |
| ----------- | ---------------------------------- | ------------------- | -------------------------------------------- | --------------------------- | ------------------------ | --------------------------- |
| `G`         | `"G - All Ages"`                   | `"G"` (6,727)       | —                                            | —                           | —                        | On Your Mark (1047)         |
| `PG`        | `"PG - Children"`                  | `"PG"` (10,394)     | —                                            | —                           | —                        | Pokemon (527), One Piece    |
| `PG13`      | `"PG-13 - Teens 13 or older"`      | —                   | —                                            | —                           | —                        | One Piece (21)              |
| `R17`       | `"R - 17+ (violence & profanity)"` | `"R"` (2,497)       | —                                            | —                           | —                        | Cowboy Bebop (1), AoT       |
| `R_PLUS`    | `"R+ - Mild Nudity"`               | —                   | —                                            | —                           | —                        | MAL 19, 26, 32              |
| `RX`        | `"Rx - Hentai"`                    | `"R18"` (0 entries) | —                                            | —                           | —                        | MAL 188, 203                |
| _(omit)_    | `null`                             | `null`              | —                                            | —                           | —                        | Some upcoming/obscure anime |
| _(numeric)_ | —                                  | —                   | `permanent` / `temporary` / `review` ratings | `aggregateRating` (JSON-LD) | community rating (score) | One Piece (69)              |

**MAL**: `AnimeRating` enum uses full Jikan strings — no normalization needed.
**AniList**: Uses `isAdult` (boolean) instead of rating.
**Kitsu**: 3 active values (G, PG, R) + R18 (0 entries). `ageRatingGuide` has free-text description. No PG-13 equivalent — Kitsu maps PG-13 content to PG with guide text.
**AnimSchedule**: No rating/age rating field.
**AniDB**: No age rating field. Has numeric community ratings in 3 categories: `permanent` (weighted, long-standing voters), `temporary` (recent/low-vote-count), `review` (from written reviews). Each has `value` (float, scale 1–10) and `count` (int). These map to `statistics["anidb"]` in the canonical model, not to `rating` (age classification). Example: One Piece permanent=8.33 (9547 votes), temporary=8.58 (10282), review=8.68 (20).
**AnimePlanet**: No age rating field (`contentRating` in JSON-LD is always `null`). Has community ratings via JSON-LD `aggregateRating`: `ratingValue` (float, scale 0.5–5), `ratingCount`, `reviewCount`, `bestRating: 5`, `worstRating: 0.5`. Maps to `statistics["anime_planet"]`. Example: One Piece ratingValue=4.314, ratingCount=63900, reviewCount=267.
**AniSearch**: No age rating field. Has community rating score (numeric, displayed as star rating on the page). Maps to `statistics["anisearch"]`, not to `rating` (age classification).

---
