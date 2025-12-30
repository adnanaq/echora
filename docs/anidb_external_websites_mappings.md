# AniDB Resource Type Mappings

This document outlines the mapping of numerical `type` attributes found in AniDB XML data to their corresponding external websites, services, and domains. 

---

## Anime-Level External Resources

This information is crucial for interpreting the `<resources>` tag for an entire anime series. The `identifier` found within each resource tag is the unique key used to locate the specific anime on that external service. It is typically used as part of the URL path or as a query string parameter.

| Type ID | Domain / Service      | Example URL Structure                                       |
|:--------|:----------------------|:------------------------------------------------------------|
| 1       | allcinema.net         | `http://www.allcinema.net/prog/show_c.php?num_c={id}`         |
| 2       | Anime News Network    | `https://www.animenewsnetwork.com/encyclopedia/anime.php?id={id}` |
| 3       | AnimeNfo              | `https://www.animenfo.com/animetitle,{id},{slug},{name}.html` |
| 4       | Official Website      | The full URL is provided directly in a `<url>` tag.         |
| 6       | Wikipedia (English)   | `https://en.wikipedia.org/wiki/{identifier}`                |
| 7       | Wikipedia (Japanese)  | `https://ja.wikipedia.org/wiki/{identifier}`                |
| 8       | Syoboi Calendar       | `http://cal.syoboi.jp/tid/{id}`                             |
| 9       | MyAnimeList           | `https://myanimelist.net/anime/{id}`                        |
| 10      | Anime-Planet          | `https://www.anime-planet.com/anime/{slug-or-id}`           |
| 26      | YouTube               | `https://www.youtube.com/{identifier}`                      |
| 28      | Crunchyroll           | `https://www.crunchyroll.com/series/{id}`                   |
| 31      | Funimation            | (Platform merged with Crunchyroll)                          |
| 32      | Amazon                | `https://www.amazon.com/dp/{asin}`                          |
| 41      | Netflix               | `https://www.netflix.com/title/{id}`                        |
| 43      | IMDb                  | `https://www.imdb.com/title/{id}`                           |
| 44      | TheMovieDB.org (TMDB) | `https://www.themoviedb.org/{type}/{id}`                    |
| 45      | Hulu                  | `https://www.hulu.com/series/{slug}`                        |

---

## Episode Number (`<epno>`) Types

Within each `<episode>` tag, the `<epno>` tag contains a `type` attribute that specifies the category of that episode.

| Type ID | Category            |
|:--------|:--------------------|
| 1       | Regular Episode     |
| 2       | Special             |
| 3       | Credit (OP/ED)      |
| 4       | Trailer             |
| 5       | Parody / Other      |