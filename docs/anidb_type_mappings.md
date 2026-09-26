# AniDB Type Mappings

This document outlines the various numerical `type` or `id` attributes found in AniDB XML data and their corresponding meanings. These mappings are essential for correctly interpreting and transforming AniDB's raw data.

---

## External Resource Types (Anime-Level)

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

### Corrections

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

### Types deliberately left unmapped

The mapper emits nothing for these rather than guessing a url. A wrong template
is worse than a missing one: type 45's Hulu entry turned Funimation identifiers
into `hulu.com` addresses that never existed.

| Type | Reason |
|:-----|:-------|
| 3 AnimeNfo | Site offline |
| 15 Marumegane | Domain is parked — serves a redirect to `/lander`, no content |
| 31 Media Arts Database | Service identified, but the database moved from `mediaarts-db.bunka.go.jp` to `mediaarts-db.artmuseums.go.jp` and the stored ids no longer resolve: the old host redirects to its root, and the new host renders an empty record for ids 12519, 2011 and 7708 |
| 12 Generasia, 13 VGMdb, 18 TV Tropes | AniDB records the type but renders no link, and animetosho has no url form either |

### Still unconfirmed

Types 17, 22 and 48 are mapped from animetosho's table but were not seen in the
133 anime sampled, so their templates remain unverified. They stay in the mapper
because nothing suggests they are wrong — unlike 15 and 31, which were tested and
failed.

---

## Character Types

Found in the `<character>` tag's `charactertype` attribute (or `character_type_id` in parsed JSON).

| Type ID | Category     | Description                                               |
|:--------|:-------------|:----------------------------------------------------------|
| 1       | Character    | Standard individual personas (e.g., Luffy, Zoro).         |
| 2       | Mecha        | Robots or mechanical entities.                            |
| 3       | Organization | Groups, crews, or collectives (e.g., Straw Hat Pirates).  |
| 4       | Vessel       | Ships or primary transportation (e.g., Going Merry).      |

---

## Episode Number (`<epno>`) Types

Found in the `<episode>` tag's `<epno>` element `type` attribute.

| Type ID | Category            |
|:--------|:--------------------|
| 1       | Regular Episode     |
| 2       | Special             |
| 3       | Credit (OP/ED)      |
| 4       | Trailer             |
| 5       | Parody / Other      |
