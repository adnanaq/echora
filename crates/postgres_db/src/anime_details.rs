//! Persistence helpers for work names, classification, themes, and images.

use std::collections::{BTreeMap, BTreeSet};

use sqlx::{Postgres, Transaction};
use uuid::Uuid;

use crate::error::Error;

/// Localized title entry stored in `anime_title`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeTitleInput {
    /// BCP 47 language code (for example `"en"` or `"ja"`).
    pub language: String,
    /// Localized work title in that language.
    pub title: String,
}

/// Theme entry stored in `anime_theme`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeThemeInput {
    /// Theme name (for example `"Psychological"`).
    pub name: String,
    /// Optional theme description.
    pub description: Option<String>,
}

/// Image URL entry stored in `anime_image`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimeImageInput {
    /// Canonical `anime_image_type` lookup value (`"cover"`, `"poster"`, or `"banner"`).
    pub image_type: String,
    /// Image URL.
    pub url: String,
}

/// Work names, classification lists, themes, and images saved with a work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnimeDetailsInput {
    /// Localized titles keyed by BCP 47 language code.
    pub titles: Vec<AnimeTitleInput>,
    /// Alternative titles and synonyms.
    pub synonyms: Vec<String>,
    /// Genre names linked via `genre`.
    pub genres: Vec<String>,
    /// Tag names linked via `tag`.
    pub tags: Vec<String>,
    /// Target demographic names linked via `demographic`.
    pub demographics: Vec<String>,
    /// Content warning names linked via `content_warning`.
    pub content_warnings: Vec<String>,
    /// Work themes with optional descriptions.
    pub themes: Vec<AnimeThemeInput>,
    /// Cover, poster, and banner image URLs.
    pub images: Vec<AnimeImageInput>,
}

/// Validates localized titles, classification lists, themes, and images.
///
/// # Errors
///
/// Returns an invalid-argument error when any required string is blank,
/// duplicate title languages exist, or an image type is unrecognized.
pub(crate) fn validate_anime_details(details: &AnimeDetailsInput) -> Result<(), Error> {
    for title_entry in &details.titles {
        if title_entry.language.trim().is_empty() || title_entry.title.trim().is_empty() {
            return Err(Error::invalid_argument(
                "anime_title entries require non-empty language and title",
            ));
        }
    }

    validate_non_empty_names(&details.synonyms, "synonym")?;
    validate_non_empty_names(&details.genres, "genre")?;
    validate_non_empty_names(&details.tags, "tag")?;
    validate_non_empty_names(&details.demographics, "demographic")?;
    validate_non_empty_names(&details.content_warnings, "content_warning")?;

    for theme_entry in &details.themes {
        if theme_entry.name.trim().is_empty() {
            return Err(Error::invalid_argument(
                "anime_theme entries require a non-empty name",
            ));
        }
    }

    for image_entry in &details.images {
        if !matches!(
            image_entry.image_type.as_str(),
            "cover" | "poster" | "banner"
        ) {
            return Err(Error::invalid_argument(format!(
                "unknown anime_image_type {:?}",
                image_entry.image_type
            )));
        }
        if image_entry.url.trim().is_empty() {
            return Err(Error::invalid_argument(
                "anime_image entries require a non-empty url",
            ));
        }
    }

    Ok(())
}

/// Checks that every name in `names` has text.
///
/// # Errors
///
/// Returns an invalid-argument error naming `field_name` when a name is blank.
fn validate_non_empty_names(names: &[String], field_name: &str) -> Result<(), Error> {
    if names.iter().any(|name| name.trim().is_empty()) {
        return Err(Error::invalid_argument(format!(
            "{field_name} entries must be non-empty"
        )));
    }
    Ok(())
}

/// Replaces all detail child tables for `anime_id` within an open transaction.
///
/// Upserts shared lookup rows in sorted order to avoid multi-worker deadlocks.
///
/// # Errors
///
/// Returns a database error when any upsert, delete, or batch insert fails.
pub(crate) async fn replace_anime_details(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    details: &AnimeDetailsInput,
) -> Result<(), Error> {
    let sorted_genres = sort_unique_strings(&details.genres);
    let sorted_tags = sort_unique_strings(&details.tags);
    let sorted_demographics = sort_unique_strings(&details.demographics);
    let sorted_content_warnings = sort_unique_strings(&details.content_warnings);

    // Upsert shared name tables in fixed, sorted key order to avoid deadlocks.
    upsert_shared_genres(tx, &sorted_genres).await?;
    upsert_shared_tags(tx, &sorted_tags).await?;
    upsert_shared_demographics(tx, &sorted_demographics).await?;
    upsert_shared_content_warnings(tx, &sorted_content_warnings).await?;

    replace_anime_titles(tx, anime_id, &details.titles).await?;
    replace_anime_synonyms(tx, anime_id, &sort_unique_strings(&details.synonyms)).await?;
    replace_anime_genres(tx, anime_id, &sorted_genres).await?;
    replace_anime_tags(tx, anime_id, &sorted_tags).await?;
    replace_anime_demographics(tx, anime_id, &sorted_demographics).await?;
    replace_anime_content_warnings(tx, anime_id, &sorted_content_warnings).await?;
    replace_anime_themes(tx, anime_id, &details.themes).await?;
    replace_anime_images(tx, anime_id, &details.images).await?;

    Ok(())
}

/// Returns the distinct values of `values` in sorted order.
fn sort_unique_strings(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Adds missing names to the shared `genre` table.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_shared_genres(
    tx: &mut Transaction<'_, Postgres>,
    names: &[String],
) -> Result<(), Error> {
    if names.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO genre (name) SELECT * FROM UNNEST($1::text[]) ON CONFLICT DO NOTHING",
        names
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Adds missing names to the shared `tag` table.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_shared_tags(
    tx: &mut Transaction<'_, Postgres>,
    names: &[String],
) -> Result<(), Error> {
    if names.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO tag (name) SELECT * FROM UNNEST($1::text[]) ON CONFLICT DO NOTHING",
        names
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Adds missing names to the shared `demographic` table.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_shared_demographics(
    tx: &mut Transaction<'_, Postgres>,
    names: &[String],
) -> Result<(), Error> {
    if names.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO demographic (name) SELECT * FROM UNNEST($1::text[]) ON CONFLICT DO NOTHING",
        names
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Adds missing names to the shared `content_warning` table.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn upsert_shared_content_warnings(
    tx: &mut Transaction<'_, Postgres>,
    names: &[String],
) -> Result<(), Error> {
    if names.is_empty() {
        return Ok(());
    }
    sqlx::query!(
        "INSERT INTO content_warning (name) SELECT * FROM UNNEST($1::text[]) ON CONFLICT DO NOTHING",
        names
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's localized titles.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_titles(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    titles: &[AnimeTitleInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_title WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let deduplicated_titles: BTreeMap<&str, &str> = titles
        .iter()
        .map(|entry| (entry.language.as_str(), entry.title.as_str()))
        .collect();
    if deduplicated_titles.is_empty() {
        return Ok(());
    }

    let languages: Vec<String> = deduplicated_titles
        .keys()
        .copied()
        .map(str::to_owned)
        .collect();
    let title_values: Vec<String> = deduplicated_titles
        .values()
        .copied()
        .map(str::to_owned)
        .collect();
    sqlx::query!(
        "INSERT INTO anime_title (anime_id, language, title)
         SELECT $1, language, title
         FROM UNNEST($2::text[], $3::text[]) AS entry(language, title)",
        anime_id,
        &languages,
        &title_values
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's synonyms.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_synonyms(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    synonyms: &[String],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_synonym WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;
    if synonyms.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO anime_synonym (anime_id, synonym)
         SELECT $1, synonym
         FROM UNNEST($2::text[]) AS entry(synonym)",
        anime_id,
        synonyms
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's genre links.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_genres(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    genres: &[String],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_genre WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;
    if genres.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO anime_genre (anime_id, genre)
         SELECT $1, genre
         FROM UNNEST($2::text[]) AS entry(genre)",
        anime_id,
        genres
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's tag links.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_tags(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    tags: &[String],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_tag WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;
    if tags.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO anime_tag (anime_id, tag)
         SELECT $1, tag
         FROM UNNEST($2::text[]) AS entry(tag)",
        anime_id,
        tags
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's demographic links.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_demographics(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    demographics: &[String],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM anime_demographic WHERE anime_id = $1",
        anime_id
    )
    .execute(&mut **tx)
    .await?;
    if demographics.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO anime_demographic (anime_id, demographic)
         SELECT $1, demographic
         FROM UNNEST($2::text[]) AS entry(demographic)",
        anime_id,
        demographics
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's content warning links.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_content_warnings(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    content_warnings: &[String],
) -> Result<(), Error> {
    sqlx::query!(
        "DELETE FROM anime_content_warning WHERE anime_id = $1",
        anime_id
    )
    .execute(&mut **tx)
    .await?;
    if content_warnings.is_empty() {
        return Ok(());
    }

    sqlx::query!(
        "INSERT INTO anime_content_warning (anime_id, content_warning)
         SELECT $1, content_warning
         FROM UNNEST($2::text[]) AS entry(content_warning)",
        anime_id,
        content_warnings
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's themes and their descriptions.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_themes(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    themes: &[AnimeThemeInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_theme WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let deduplicated_themes: BTreeMap<&str, Option<&str>> = themes
        .iter()
        .map(|entry| (entry.name.as_str(), entry.description.as_deref()))
        .collect();
    if deduplicated_themes.is_empty() {
        return Ok(());
    }

    let names: Vec<String> = deduplicated_themes
        .keys()
        .copied()
        .map(str::to_owned)
        .collect();
    let descriptions: Vec<Option<String>> = deduplicated_themes
        .values()
        .map(|description| description.map(str::to_owned))
        .collect();
    sqlx::query!(
        "INSERT INTO anime_theme (anime_id, name, description)
         SELECT $1, name, description
         FROM UNNEST($2::text[], $3::text[]) AS entry(name, description)",
        anime_id,
        &names,
        &descriptions as &[Option<String>]
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Replaces a work's cover, poster and banner images.
///
/// # Errors
///
/// Returns a database error when a query fails.
async fn replace_anime_images(
    tx: &mut Transaction<'_, Postgres>,
    anime_id: Uuid,
    images: &[AnimeImageInput],
) -> Result<(), Error> {
    sqlx::query!("DELETE FROM anime_image WHERE anime_id = $1", anime_id)
        .execute(&mut **tx)
        .await?;

    let deduplicated_images: BTreeSet<(&str, &str)> = images
        .iter()
        .map(|entry| (entry.image_type.as_str(), entry.url.as_str()))
        .collect();
    if deduplicated_images.is_empty() {
        return Ok(());
    }

    let image_types: Vec<String> = deduplicated_images
        .iter()
        .map(|(image_type, _)| (*image_type).to_owned())
        .collect();
    let urls: Vec<String> = deduplicated_images
        .iter()
        .map(|(_, url)| (*url).to_owned())
        .collect();
    sqlx::query!(
        "INSERT INTO anime_image (anime_id, image_type, source)
         SELECT $1, image_type, source
         FROM UNNEST($2::text[], $3::text[]) AS entry(image_type, source)",
        anime_id,
        &image_types,
        &urls
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    #[test]
    fn validate_anime_details_checks_all_fields() {
        let valid = AnimeDetailsInput {
            titles: vec![
                AnimeTitleInput {
                    language: "en".to_owned(),
                    title: "Cowboy Bebop".to_owned(),
                },
                AnimeTitleInput {
                    language: "ja".to_owned(),
                    title: "カウボーイビバップ".to_owned(),
                },
            ],
            synonyms: vec!["Bebop".to_owned()],
            genres: vec!["Action".to_owned(), "Sci-Fi".to_owned()],
            tags: vec!["Space".to_owned()],
            demographics: vec!["Seinen".to_owned()],
            content_warnings: vec!["Violence".to_owned()],
            themes: vec![AnimeThemeInput {
                name: "Space Western".to_owned(),
                description: Some("Jazz and bounty hunting.".to_owned()),
            }],
            images: vec![
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://example.com/cover.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "poster".to_owned(),
                    url: "https://example.com/poster.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "banner".to_owned(),
                    url: "https://example.com/banner.jpg".to_owned(),
                },
            ],
        };
        assert!(validate_anime_details(&valid).is_ok());

        let mut blank_lang = valid.clone();
        blank_lang.titles[0].language = "  ".to_owned();
        assert!(matches!(validate_anime_details(&blank_lang), Err(e) if e.is_invalid_argument()));

        let mut blank_title = valid.clone();
        blank_title.titles[0].title = String::new();
        assert!(matches!(validate_anime_details(&blank_title), Err(e) if e.is_invalid_argument()));

        for mutate in [
            |details: &mut AnimeDetailsInput| details.synonyms = vec![" ".to_owned()],
            |details: &mut AnimeDetailsInput| details.genres = vec![String::new()],
            |details: &mut AnimeDetailsInput| details.tags = vec!["\t".to_owned()],
            |details: &mut AnimeDetailsInput| details.demographics = vec![String::new()],
            |details: &mut AnimeDetailsInput| details.content_warnings = vec![" ".to_owned()],
        ] {
            let mut bad = valid.clone();
            mutate(&mut bad);
            assert!(matches!(validate_anime_details(&bad), Err(e) if e.is_invalid_argument()));
        }

        let mut blank_theme = valid.clone();
        blank_theme.themes[0].name = " ".to_owned();
        assert!(matches!(validate_anime_details(&blank_theme), Err(e) if e.is_invalid_argument()));

        let mut bad_image_type = valid.clone();
        bad_image_type.images[0].image_type = "thumbnail".to_owned();
        assert!(matches!(
            validate_anime_details(&bad_image_type),
            Err(e) if e.is_invalid_argument()
        ));

        let mut blank_image_url = valid;
        blank_image_url.images[0].url = "   ".to_owned();
        assert!(matches!(
            validate_anime_details(&blank_image_url),
            Err(e) if e.is_invalid_argument()
        ));
    }

    #[test]
    fn sort_unique_strings_deduplicates_and_sorts() {
        let input = vec![
            "Sci-Fi".to_owned(),
            "Action".to_owned(),
            "Sci-Fi".to_owned(),
            "Drama".to_owned(),
        ];
        assert_eq!(
            sort_unique_strings(&input),
            vec!["Action".to_owned(), "Drama".to_owned(), "Sci-Fi".to_owned()]
        );
    }

    #[sqlx::test]
    async fn replace_anime_details_inserts_and_replaces_all_tables(pool: PgPool) {
        let anime_id = Uuid::now_v7();
        sqlx::query!(
            "INSERT INTO anime (id, title, type, status, rating)
             VALUES ($1, 'Cowboy Bebop', 'TV', 'FINISHED', 'R - 17+ (violence & profanity)')",
            anime_id,
        )
        .execute(&pool)
        .await
        .unwrap();

        let populated = AnimeDetailsInput {
            titles: vec![
                AnimeTitleInput {
                    language: "en".to_owned(),
                    title: "Cowboy Bebop".to_owned(),
                },
                AnimeTitleInput {
                    language: "ja".to_owned(),
                    title: "カウボーイビバップ".to_owned(),
                },
            ],
            synonyms: vec!["Bebop".to_owned(), "Bebop".to_owned()],
            genres: vec!["Sci-Fi".to_owned(), "Action".to_owned()],
            tags: vec!["Space".to_owned()],
            demographics: vec!["Seinen".to_owned()],
            content_warnings: vec!["Violence".to_owned()],
            themes: vec![AnimeThemeInput {
                name: "Bounty Hunters".to_owned(),
                description: Some("2071 solar system.".to_owned()),
            }],
            images: vec![
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://example.com/cover.jpg".to_owned(),
                },
                AnimeImageInput {
                    image_type: "cover".to_owned(),
                    url: "https://example.com/cover.jpg".to_owned(),
                },
            ],
        };

        let mut tx = pool.begin().await.unwrap();
        replace_anime_details(&mut tx, anime_id, &populated)
            .await
            .unwrap();
        tx.commit().await.unwrap();

        let title_count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM anime_title WHERE anime_id = $1",
            anime_id
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(title_count, Some(2));

        let genres: Vec<String> = sqlx::query_scalar!(
            "SELECT genre FROM anime_genre WHERE anime_id = $1 ORDER BY genre",
            anime_id
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(genres, vec!["Action".to_owned(), "Sci-Fi".to_owned()]);

        let image_count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM anime_image WHERE anime_id = $1",
            anime_id
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(image_count, Some(1));

        let mut tx = pool.begin().await.unwrap();
        replace_anime_details(&mut tx, anime_id, &AnimeDetailsInput::default())
            .await
            .unwrap();
        tx.commit().await.unwrap();

        let remaining_genres = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM anime_genre WHERE anime_id = $1",
            anime_id
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(remaining_genres, Some(0));
    }
}
