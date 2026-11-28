//! External sites whose anime pages the catalogue stores.

use std::fmt;
use std::str::FromStr;

use crate::{EntityType, KeyError};

/// Supported external anime metadata providers.
///
/// The variant order is the order [`Ord`] sorts providers by. [`Provider::ALL`] lists them
/// in the order of the `provider` lookup table instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(try_from = "String", into = "&'static str")
)]
pub enum Provider {
    /// `MyAnimeList` (`"mal"`).
    Mal,
    /// `AniList` (`"anilist"`).
    AniList,
    /// `AniDB` (`"anidb"`).
    AniDb,
    /// aniSearch (`"anisearch"`).
    AniSearch,
    /// Anime News Network (`"ann"`).
    Ann,
    /// LiveChart.me (`"livechart"`).
    LiveChart,
    /// SIMKL (`"simkl"`).
    Simkl,
    /// `AnimeCountdown` (`"animecountdown"`).
    AnimeCountdown,
    /// Syoboi Calendar (`"syoboi"`).
    Syoboi,
    /// Kitsu (`"kitsu"`).
    Kitsu,
    /// Anime-Planet (`"anime_planet"`).
    AnimePlanet,
    /// AnimeSchedule.net (`"animeschedule"`).
    AnimeSchedule,
}

impl Provider {
    /// Every provider, in the order the `provider` lookup table lists them.
    pub const ALL: [Self; 12] = [
        Self::Mal,
        Self::AniList,
        Self::Kitsu,
        Self::AniDb,
        Self::AnimePlanet,
        Self::AniSearch,
        Self::AnimeSchedule,
        Self::Syoboi,
        Self::Ann,
        Self::LiveChart,
        Self::Simkl,
        Self::AnimeCountdown,
    ];

    /// Returns the canonical database key for this provider.
    ///
    /// The same name is used on the wire, in Temporal payloads and in task queue names.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::Provider;
    ///
    /// assert_eq!(Provider::Mal.as_str(), "mal");
    /// assert_eq!(Provider::AnimePlanet.as_str(), "anime_planet");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mal => "mal",
            Self::AniList => "anilist",
            Self::AniDb => "anidb",
            Self::AniSearch => "anisearch",
            Self::Ann => "ann",
            Self::LiveChart => "livechart",
            Self::Simkl => "simkl",
            Self::AnimeCountdown => "animecountdown",
            Self::Syoboi => "syoboi",
            Self::Kitsu => "kitsu",
            Self::AnimePlanet => "anime_planet",
            Self::AnimeSchedule => "animeschedule",
        }
    }

    /// Validates that `address` matches this provider's address format for `entity_type`.
    ///
    /// A MAL episode page has no ID of its own, so its address is the work's ID and the
    /// episode number joined by a colon, as in `21:1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use echora_domain::{EntityType, Provider};
    ///
    /// assert!(Provider::Mal.validate_address(EntityType::Anime, "21").is_ok());
    /// assert!(Provider::Mal.validate_address(EntityType::Episode, "21:1").is_ok());
    /// assert!(Provider::Kitsu.validate_address(EntityType::Anime, "cowboy-bebop").is_ok());
    /// assert!(Provider::Mal.validate_address(EntityType::Anime, "not-numeric").is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an invalid-address error when `address` is empty or violates
    /// the provider's numeric or slug constraints.
    pub fn validate_address(
        self,
        entity_type: EntityType,
        address: impl AsRef<str>,
    ) -> Result<(), KeyError> {
        let address = address.as_ref();
        if address.is_empty() {
            return Err(KeyError::invalid_address("page address must not be empty"));
        }

        match self {
            Self::Mal if entity_type == EntityType::Episode => {
                let is_work_and_episode_number =
                    address
                        .split_once(':')
                        .is_some_and(|(work_address, episode_number)| {
                            is_numeric_address(work_address) && is_numeric_address(episode_number)
                        });
                if is_work_and_episode_number {
                    Ok(())
                } else {
                    Err(KeyError::invalid_address(format!(
                        "provider mal requires an episode address of the form \"<work id>:<episode number>\", got {address:?}"
                    )))
                }
            }
            Self::Mal
            | Self::AniList
            | Self::AniDb
            | Self::AniSearch
            | Self::Ann
            | Self::LiveChart
            | Self::Simkl
            | Self::AnimeCountdown
            | Self::Syoboi => {
                if is_numeric_address(address) {
                    Ok(())
                } else {
                    Err(KeyError::invalid_address(format!(
                        "provider {} requires a numeric address, got {address:?}",
                        self.as_str()
                    )))
                }
            }
            Self::Kitsu | Self::AnimePlanet | Self::AnimeSchedule => {
                if address
                    .bytes()
                    .any(|byte| matches!(byte, b'/' | b'?' | b'#' | b' ' | b'\t' | b'\n' | b'\r'))
                {
                    Err(KeyError::invalid_address(format!(
                        "provider {} address contains invalid path or whitespace characters: {address:?}",
                        self.as_str()
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Returns whether `address` is a non-empty run of ASCII digits.
fn is_numeric_address(address: &str) -> bool {
    !address.is_empty() && address.bytes().all(|byte| byte.is_ascii_digit())
}

impl AsRef<str> for Provider {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Provider> for &'static str {
    fn from(provider: Provider) -> Self {
        provider.as_str()
    }
}

impl FromStr for Provider {
    type Err = KeyError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "mal" => Ok(Self::Mal),
            "anilist" => Ok(Self::AniList),
            "anidb" => Ok(Self::AniDb),
            "anisearch" => Ok(Self::AniSearch),
            "ann" => Ok(Self::Ann),
            "livechart" => Ok(Self::LiveChart),
            "simkl" => Ok(Self::Simkl),
            "animecountdown" => Ok(Self::AnimeCountdown),
            "syoboi" => Ok(Self::Syoboi),
            "kitsu" => Ok(Self::Kitsu),
            "anime_planet" => Ok(Self::AnimePlanet),
            "animeschedule" => Ok(Self::AnimeSchedule),
            _ => Err(KeyError::unknown_provider(value)),
        }
    }
}

impl TryFrom<String> for Provider {
    type Error = KeyError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_validates_numeric_and_slug_address() {
        assert!(
            Provider::Mal
                .validate_address(EntityType::Anime, "21")
                .is_ok()
        );
        assert!(
            Provider::AniDb
                .validate_address(EntityType::Episode, "12345")
                .is_ok()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Anime, "21extra")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Anime, "21:1")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21:1")
                .is_ok()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21:")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, ":1")
                .is_err()
        );
        assert!(
            Provider::Mal
                .validate_address(EntityType::Episode, "21:1:2")
                .is_err()
        );
        assert!(
            Provider::Kitsu
                .validate_address(EntityType::Anime, "cowboy-bebop")
                .is_ok()
        );
        assert!(
            Provider::Kitsu
                .validate_address(EntityType::Anime, "bad/slug")
                .is_err()
        );
        assert!(
            Provider::Kitsu
                .validate_address(EntityType::Anime, "")
                .is_err()
        );
        assert!("unknown_provider".parse::<Provider>().is_err());
    }

    #[test]
    fn provider_from_str_round_trips_every_lookup_value() {
        for provider_name in [
            "mal",
            "anilist",
            "anidb",
            "anisearch",
            "ann",
            "livechart",
            "simkl",
            "animecountdown",
            "syoboi",
            "kitsu",
            "anime_planet",
            "animeschedule",
        ] {
            let provider: Provider = provider_name.parse().unwrap();
            assert_eq!(provider.as_str(), provider_name);
            assert_eq!(provider.to_string(), provider_name);
            assert_eq!(provider.as_ref(), provider_name);
            assert_eq!(<&'static str>::from(provider), provider_name);
            assert_eq!(
                Provider::try_from(provider_name.to_owned()).ok(),
                Some(provider)
            );
        }
    }

    #[test]
    fn provider_all_lists_every_provider_once() {
        let mut providers = Provider::ALL.to_vec();
        providers.sort_unstable();
        providers.dedup();
        assert_eq!(providers.len(), Provider::ALL.len());
        for provider in Provider::ALL {
            assert_eq!(provider.as_str().parse::<Provider>().ok(), Some(provider));
        }
    }

    #[test]
    fn provider_from_str_unknown_name_returns_unknown_provider() {
        let error = "myanimelist".parse::<Provider>().unwrap_err();
        assert!(error.is_unknown_provider());
        assert!(!error.is_unknown_entity_type());
        assert!(!error.is_invalid_address());
        assert!(
            error
                .to_string()
                .starts_with("unknown provider \"myanimelist\"")
        );
    }
}
