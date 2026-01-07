# AniDB Type Mappings

This document outlines the various numerical `type` or `id` attributes found in AniDB XML data and their corresponding meanings. These mappings are essential for correctly interpreting and transforming AniDB's raw data.

---

## External Resource Types (Anime-Level)

Found in the `<resources>` tag for an entire anime series. The `identifier` is the unique key used on the external service.

| Type ID | Domain / Service      | Example URL Structure                                       | Verified |
|:--------|:----------------------|:------------------------------------------------------------|:---------|
| 1       | Anime News Network    | `https://www.animenewsnetwork.com/encyclopedia/anime.php?id={id}` | ✅ |
| 2       | MyAnimeList           | `https://myanimelist.net/anime/{id}`                        | ✅ |
| 3       | AnimeNfo              | `https://www.animenfo.com/animetitle,{id},{slug},{name}.html` | ❌ OFFLINE |
| 4       | Official Website      | The full URL is provided directly in a `<url>` tag.         | ✅ |
| 6       | Wikipedia (English)   | `https://en.wikipedia.org/wiki/{identifier}`                | ✅ |
| 7       | Wikipedia (Japanese)  | `https://ja.wikipedia.org/wiki/{identifier}`                | ✅ |
| 8       | Syoboi Calendar       | `http://cal.syoboi.jp/tid/{id}`                             | ✅ |
| 9       | allcinema.net         | `http://www.allcinema.net/prog/show_c.php?num_c={id}`       | ✅ |
| 10      | Anison.info           | `http://anison.info/data/program/{id}.html`                 | ✅ |
| 26      | YouTube               | `https://www.youtube.com/{identifier}`                      | ✅ |
| 28      | Crunchyroll           | `https://www.crunchyroll.com/watch/{id}` (episode-level)    | ✅ |
| 31      | Funimation            | (Platform merged with Crunchyroll)                          | ❌ DEPRECATED |
| 32      | Amazon                | `https://www.amazon.com/dp/{asin}`                          | ✅ |
| 41      | Netflix               | `https://www.netflix.com/title/{id}`                        | ✅ |
| 43      | IMDb                  | `https://www.imdb.com/title/{id}`                           | ✅ |
| 44      | TheMovieDB.org (TMDB) | `https://www.themoviedb.org/{type}/{id}`                    | ✅ |
| 45      | Hulu                  | `https://www.hulu.com/series/{slug}`                        | ✅ |

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
