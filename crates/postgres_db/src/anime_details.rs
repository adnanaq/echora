//! Persistence helpers for work names, classification, themes, and images.

// Rust guideline compliant 2026-10-09

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

fn validate_non_empty_names(names: &[String], field_name: &str) -> Result<(), Error> {
    if names.iter().any(|name| name.trim().is_empty()) {
        return Err(Error::invalid_argument(format!(
            "{field_name} entries must be non-empty"
        )));
    }
    Ok(())
}

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

fn sort_unique_strings(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

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
