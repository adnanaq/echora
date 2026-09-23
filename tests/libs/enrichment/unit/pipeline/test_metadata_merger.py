"""Tests for cross-provider anime metadata consolidation.

Focus is arbitration: which provider's value survives, and what counts as a
value at all. Mappers emit a key for every model field, so "absent" arrives as
an empty container rather than a missing key — the case that silently discarded
AniDB's `titles` before `_provider_supplied` existed.
"""

import pytest
from enrichment.pipeline.link_rules import merged_anime_id
from enrichment.pipeline.metadata_merger import merge_provider_records

_MAL_URL = "https://myanimelist.net/anime/21/One_Piece"


def _record(**fields: object) -> dict[str, object]:
    return {"sources": [_MAL_URL], **fields}


def test_first_signal_takes_the_most_trusted_concrete_value() -> None:
    merged = merge_provider_records(
        {
            "kitsu": _record(type="OVA", year=2001),
            "mal": _record(type="TV", year=1999),
        }
    )
    assert merged["type"] == "TV"
    assert merged["year"] == 1999


def test_sentinel_never_beats_a_concrete_value() -> None:
    # A real value from the lowest-ranked provider outranks UNKNOWN from the
    # highest, so "no data" cannot masquerade as an answer.
    merged = merge_provider_records(
        {"mal": _record(type="UNKNOWN"), "kitsu": _record(type="TV")}
    )
    assert merged["type"] == "TV"


def test_sentinel_survives_when_every_provider_agrees_on_it() -> None:
    merged = merge_provider_records(
        {"mal": _record(type="UNKNOWN"), "kitsu": _record(type="UNKNOWN")}
    )
    assert merged["type"] == "UNKNOWN"


def test_empty_container_does_not_outrank_a_populated_one() -> None:
    # Six providers emit `titles: {}`; only AniDB fills it. Treating {} as a
    # value let MAL's empty dict win and dropped all 27 titles.
    merged = merge_provider_records(
        {"mal": _record(titles={}), "anidb": _record(titles={"de": "One Piece"})}
    )
    assert merged["titles"] == {"de": "One Piece"}


def test_disagreement_on_a_uniform_field_is_logged(caplog) -> None:
    merge_provider_records({"mal": _record(year=1999), "kitsu": _record(year=2001)})
    assert "disagree on year" in caplog.text


def test_statistics_are_kept_per_provider_without_arbitration() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(statistics={"mal": {"score": 8.73}}),
            "anisearch": _record(statistics={"anisearch": {"score": 8.36}}),
        }
    )
    assert merged["statistics"] == {
        "anisearch": {"score": 8.36},
        "mal": {"score": 8.73},
    }


def test_sources_keep_one_url_per_work_preferring_the_slug() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(),
            "anilist": {"sources": ["https://myanimelist.net/anime/21"]},
        }
    )
    assert merged["sources"] == [_MAL_URL]


def test_sources_union_the_offline_seed_and_the_providers() -> None:
    # Neither side is complete: animeschedule is absent from the seed, and
    # livechart is absent from every provider.
    merged = merge_provider_records(
        {
            "animeschedule": _record(
                sources=["https://animeschedule.net/anime/one-piece"]
            )
        },
        {"sources": ["https://livechart.me/anime/321"]},
    )
    assert merged["sources"] == [
        "https://animeschedule.net/anime/one-piece",
        "https://livechart.me/anime/321",
    ]


def test_numeric_url_drops_when_a_slug_names_the_same_work() -> None:
    # Kitsu is addressable both ways and the two halves disagree on which to
    # use, so the same anime arrives as two URLs.
    merged = merge_provider_records(
        {"kitsu": _record(sources=["https://kitsu.io/anime/one-piece"])},
        {"sources": ["https://kitsu.app/anime/12"]},
    )
    assert merged["sources"] == ["https://kitsu.io/anime/one-piece"]


def test_numeric_url_survives_when_no_slug_rivals_it() -> None:
    # MAL and AniList identify every work numerically; dropping those would
    # empty `sources`.
    merged = merge_provider_records({"mal": _record()})
    assert merged["sources"] == [_MAL_URL]


def test_episode_count_reads_zero_as_untracked() -> None:
    # episode_count defaults to 0 in the model, so a provider that does not
    # track it looks identical to one reporting no episodes. AniList ranks
    # above AniDB, and its 0 must not win.
    merged = merge_provider_records(
        {"anilist": _record(episode_count=0), "anidb": _record(episode_count=1184)}
    )
    assert merged["episode_count"] == 1184


def test_episode_count_uses_its_own_provider_order() -> None:
    merged = merge_provider_records(
        {
            "anime_planet": _record(episode_count=1179),
            "anidb": _record(episode_count=1184),
            "mal": _record(episode_count=1174),
        }
    )
    assert merged["episode_count"] == 1174


def test_episode_count_is_zero_when_nobody_tracks_it() -> None:
    assert merge_provider_records({"kitsu": _record()})["episode_count"] == 0


def test_synopsis_prefers_the_longest_once_markup_is_gone() -> None:
    # MAL outranks AniSearch but says less; the attribution footers are not
    # content and must not count toward length.
    merged = merge_provider_records(
        {
            "mal": _record(synopsis="Short prose.\n\n[Written by MAL Rewrite]"),
            "anisearch": _record(
                synopsis="Much longer prose about pirates. Source: www.anisearch.com/x"
            ),
        }
    )
    assert merged["synopsis"] == "Much longer prose about pirates."


def test_synopsis_falls_back_to_priority_between_comparable_texts() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(synopsis="A" * 100),
            "anisearch": _record(synopsis="B" * 104),
        }
    )
    assert merged["synopsis"] == "A" * 100


def test_title_prefers_the_readable_casing_over_priority() -> None:
    merged = merge_provider_records(
        {"anilist": _record(title="ONE PIECE"), "kitsu": _record(title="One Piece")}
    )
    assert merged["title"] == "One Piece"


def test_title_keeps_priority_when_titles_differ_beyond_case() -> None:
    merged = merge_provider_records(
        {"mal": _record(title="ONE PIECE"), "kitsu": _record(title="Wan Pisu")}
    )
    assert merged["title"] == "ONE PIECE"


def test_title_japanese_prefers_japanese_script_over_romaji() -> None:
    # Four providers file romaji under this field; only the lower-ranked ones
    # file kana, so priority alone fills it with the wrong script.
    merged = merge_provider_records(
        {
            "mal": _record(title_japanese="ONE PIECE"),
            "anisearch": _record(title_japanese="ワンピース"),
        }
    )
    assert merged["title_japanese"] == "ワンピース"


def test_title_japanese_keeps_romaji_when_nobody_supplies_kana() -> None:
    merged = merge_provider_records({"mal": _record(title_japanese="ONE PIECE")})
    assert merged["title_japanese"] == "ONE PIECE"


def test_id_is_stable_across_differing_provider_coverage() -> None:
    full = merge_provider_records(
        {
            "mal": _record(),
            "kitsu": _record(sources=["https://kitsu.io/anime/one-piece"]),
        }
    )
    mal_only = merge_provider_records({"mal": _record()})
    assert full["id"] == mal_only["id"]


def test_id_ignores_slug_decoration() -> None:
    assert merged_anime_id([_MAL_URL]) == merged_anime_id(
        ["https://myanimelist.net/anime/21"]
    )


def test_id_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="Cannot derive an anime id"):
        merged_anime_id([])


def test_the_most_trusted_provider_decides_the_field() -> None:
    # Shounen arrives as a demographic, a genre and a theme depending on who
    # is asked. MAL is trusted most, so its demographic stands - AnimeSchedule
    # calling it a genre does not override that.
    merged = merge_provider_records(
        {
            "mal": _record(demographics=["Shounen"]),
            "animeschedule": _record(genres=["Shounen", "Action"]),
            "kitsu": _record(themes=[{"name": "Shounen"}, {"name": "Super Power"}]),
        }
    )
    assert merged["demographics"] == ["Shounen"]
    assert merged["genres"] == ["Action"]
    assert [t["name"] for t in merged["themes"]] == ["Super Power"]


def test_a_tag_is_not_a_classification() -> None:
    # AniDB has no genre, theme or demographic field, so everything it knows
    # arrives as a tag. Letting that count would demote the word on its
    # say-so alone, even though a less trusted provider classified it.
    merged = merge_provider_records(
        {
            "anidb": _record(tags=["swordplay"]),
            "kitsu": _record(themes=[{"name": "Swordplay"}]),
        }
    )
    assert [t["name"] for t in merged["themes"]] == ["Swordplay"]
    assert merged["tags"] == []


def test_field_order_settles_one_provider_using_two_of_its_own_fields() -> None:
    # Kitsu files the same word under both its genres and its themes.
    merged = merge_provider_records(
        {"kitsu": _record(genres=["Super Power"], themes=[{"name": "Super Power"}])}
    )
    assert merged["genres"] == ["Super Power"]
    assert merged["themes"] == []


def test_tags_only_hold_what_nobody_else_claimed() -> None:
    merged = merge_provider_records(
        {
            "anidb": _record(tags=["Military", "bounty hunter"]),
            "mal": _record(themes=[{"name": "Military"}]),
        }
    )
    assert [t["name"] for t in merged["themes"]] == ["Military"]
    assert merged["tags"] == ["bounty hunter"]


def test_a_word_no_provider_classified_stays_a_tag() -> None:
    # AniDB files everything as a tag. With nobody else supplying the word,
    # a tag is what it is - we do not second-guess it from a word list.
    merged = merge_provider_records({"anidb": _record(tags=["ecchi"])})
    assert merged["tags"] == ["ecchi"]
    assert merged["genres"] == []


def test_provider_specific_genres_are_kept_as_genres() -> None:
    # AniSearch has genres MAL never heard of. They are still genres.
    merged = merge_provider_records(
        {"anisearch": _record(genres=["Fighting-Shounen", "Ganbatte"])}
    )
    assert merged["genres"] == ["Fighting-Shounen", "Ganbatte"]


def test_most_providers_decide_the_spelling() -> None:
    # AniDB writes everything lowercase. One provider must not decide how a
    # word is stored when six others disagree.
    merged = merge_provider_records(
        {
            "mal": _record(genres=["Action"]),
            "anilist": _record(genres=["Action"]),
            "anidb": _record(genres=["action"]),
        }
    )
    assert merged["genres"] == ["Action"]


def test_the_trusted_provider_breaks_a_spelling_tie() -> None:
    merged = merge_provider_records(
        {"mal": _record(genres=["Sci-Fi"]), "anidb": _record(genres=["sci-fi"])}
    )
    assert merged["genres"] == ["Sci-Fi"]


def test_differently_written_words_collapse_into_one() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(genres=["Sci-Fi"]),
            "anidb": _record(tags=["science fiction"]),
            "kitsu": _record(themes=[{"name": "Cross Dressing"}]),
            "anilist": _record(tags=["Crossdressing"]),
        }
    )
    assert merged["genres"] == ["Sci-Fi"]
    assert "science fiction" not in merged["tags"]
    assert [t["name"] for t in merged["themes"]] == ["Cross Dressing"]


def test_a_theme_keeps_its_description() -> None:
    merged = merge_provider_records(
        {"anilist": _record(themes=[{"name": "Travel", "description": "Moves about."}])}
    )
    assert merged["themes"] == [{"name": "Travel", "description": "Moves about."}]


def test_a_word_promoted_out_of_themes_drops_the_description() -> None:
    # genres hold plain strings, so a word leaving themes has to change shape.
    merged = merge_provider_records(
        {
            "kitsu": _record(themes=[{"name": "Magic", "description": "Spells."}]),
            "animeschedule": _record(genres=["Magic"]),
        }
    )
    assert merged["genres"] == ["Magic"]
    assert merged["themes"] == []


def test_object_fields_merge_one_sub_field_at_a_time() -> None:
    # Providers partition broadcast rather than duplicating it; taking the
    # highest-ranked object whole would discard everyone else's keys.
    merged = merge_provider_records(
        {
            "mal": _record(broadcast={"day": "Sundays", "time": "23:15"}),
            "anilist": _record(broadcast={"next_episode_at": "2026-09-20T14:16:00Z"}),
            "animeschedule": _record(broadcast={"jp_time": "2025-04-01T14:15:00Z"}),
        }
    )
    assert merged["broadcast"] == {
        "day": "Sundays",
        "time": "23:15",
        "next_episode_at": "2026-09-20T14:16:00Z",
        "jp_time": "2025-04-01T14:15:00Z",
    }


def test_month_falls_back_to_the_premiere_date() -> None:
    # AnimeSchedule is the only provider that names the month, so without a
    # fallback the field empties whenever that one fetch fails.
    merged = merge_provider_records(
        {"mal": _record(aired_dates={"aired_from": "1999-10-19T15:00:00Z"})}
    )
    assert merged["month"] == "October"


def test_stated_month_beats_the_derived_one() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(aired_dates={"aired_from": "1999-10-19T15:00:00Z"}),
            "animeschedule": _record(month="October"),
        }
    )
    assert merged["month"] == "October"


def test_images_union_each_category_separately() -> None:
    # A provider with covers but no banners must not suppress another's banners.
    merged = merge_provider_records(
        {
            "mal": _record(images={"covers": ["a.jpg"], "banners": []}),
            "anilist": _record(images={"covers": ["b.jpg"], "banners": ["c.jpg"]}),
        }
    )
    assert merged["images"] == {"covers": ["a.jpg", "b.jpg"], "banners": ["c.jpg"]}


def test_streaming_entries_fold_scheme_and_platform_casing() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(
                streaming_sources=[
                    {
                        "platform": "Crunchyroll",
                        "source": "http://crunchyroll.com/one-piece",
                    }
                ]
            ),
            "animeschedule": _record(
                streaming_sources=[
                    {
                        "platform": "crunchyroll",
                        "source": "https://www.crunchyroll.com/one-piece",
                    }
                ]
            ),
        }
    )
    assert merged["streaming_sources"] == [
        {"platform": "crunchyroll", "source": "http://crunchyroll.com/one-piece"}
    ]


def test_different_channels_on_one_platform_both_survive() -> None:
    # The two One Piece YouTube channels are a Japanese and an English one.
    merged = merge_provider_records(
        {
            "mal": _record(
                streaming_sources=[
                    {
                        "platform": "YouTube",
                        "source": "https://youtube.com/@onepieceofficial",
                    },
                    {
                        "platform": "YouTube",
                        "source": "https://youtube.com/@OnePieceOfficialENG",
                    },
                ]
            )
        }
    )
    assert len(merged["streaming_sources"]) == 2


def test_trailers_keep_different_videos_and_the_richer_record() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(
                trailers=[{"source": "https://youtube.com/watch?v=aaa", "title": "PV"}]
            ),
            "kitsu": _record(
                trailers=[
                    {"source": "https://www.youtube.com/watch?v=aaa"},
                    {"source": "https://youtube.com/watch?v=bbb"},
                ]
            ),
        }
    )
    assert merged["trailers"] == [
        {"source": "https://youtube.com/watch?v=aaa", "title": "PV"},
        {"source": "https://youtube.com/watch?v=bbb"},
    ]


def test_external_sources_drop_links_another_field_owns() -> None:
    # AniDB files provider pages and streaming platforms here; the urls differ
    # from the ones sources and streaming_sources hold, so only the platform
    # reveals that they are already owned.
    merged = merge_provider_records(
        {
            "anidb": _record(
                external_sources=[
                    {
                        "platform": "anidb",
                        "source": "https://anidb.net/perl-bin/x?aid=69",
                    },
                    {
                        "platform": "crunchyroll",
                        "source": "https://crunchyroll.com/series/GRMG",
                    },
                    {
                        "platform": "wikipedia_en",
                        "source": "https://en.wikipedia.org/wiki/One_Piece",
                    },
                ]
            )
        }
    )
    assert [e["platform"] for e in merged["external_sources"]] == ["wikipedia_en"]
    # The provider page is dropped (sources already names that work), but the
    # streaming link is MOVED, not dropped — AniDB's crunchyroll url differs
    # from every other provider's, so dropping it loses the link outright.
    assert [e["source"] for e in merged["streaming_sources"]] == [
        "https://crunchyroll.com/series/GRMG"
    ]


def test_external_sources_fold_one_page_published_two_ways() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(
                external_sources=[
                    {
                        "platform": "official_site",
                        "source": "http://www.toei-anim.co.jp/tv/onep/",
                    }
                ]
            ),
            "anisearch": _record(
                external_sources=[
                    {
                        "platform": "official_site",
                        "source": "https://toei-anim.co.jp/tv/onep",
                        "language": "Japanese",
                    }
                ]
            ),
        }
    )
    assert merged["external_sources"] == [
        {
            "platform": "official_site",
            "source": "http://www.toei-anim.co.jp/tv/onep/",
            "language": "Japanese",
        }
    ]


def test_synonyms_drop_the_titles_already_resolved() -> None:
    # AniDB lists the work's own title among its synonyms.
    merged = merge_provider_records(
        {
            "mal": _record(title="One Piece"),
            "anidb": _record(synonyms=["One Piece", "ワンピース", "Wan Pisu"]),
        }
    )
    assert merged["synonyms"] == ["ワンピース", "Wan Pisu"]


def test_synonyms_fold_apostrophe_variants() -> None:
    merged = merge_provider_records(
        {
            "anidb": _record(
                synonyms=[
                    "All`arrembaggio!",
                    "All'arrembaggio!",
                    "All\u2019arrembaggio!",
                ]
            )
        }
    )
    assert merged["synonyms"] == ["All`arrembaggio!"]


def test_relations_are_merged_in_the_same_pass() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(
                related_anime={
                    "SEQUEL": [
                        {
                            "title": "One Piece Film: Red",
                            "type": "MOVIE",
                            "sources": ["https://myanimelist.net/anime/50410"],
                        }
                    ]
                }
            )
        }
    )
    assert merged["related_anime"]["SEQUEL"][0]["title"] == "One Piece Film: Red"


def test_streaming_link_filed_as_external_is_moved_not_dropped() -> None:
    # AniDB files Crunchyroll, Amazon and Funimation under external_sources at
    # urls nobody else reports. Excluding them from the residual field without
    # collecting them elsewhere loses them.
    merged = merge_provider_records(
        {
            "anidb": _record(
                external_sources=[
                    {
                        "platform": "funimation",
                        "source": "https://funimation.com/shows/one-piece",
                    }
                ]
            )
        }
    )
    assert merged["streaming_sources"] == [
        {"platform": "funimation", "source": "https://funimation.com/shows/one-piece"}
    ]
    assert "external_sources" not in merged


def test_nsfw_takes_any_provider_flag_over_priority() -> None:
    # AniList outranks Kitsu, so priority would mark a work Kitsu flags as
    # adult "safe". The two errors are not equally bad.
    merged = merge_provider_records(
        {"anilist": _record(nsfw=False), "kitsu": _record(nsfw=True)}
    )
    assert merged["nsfw"] is True


def test_nsfw_is_false_only_when_no_provider_flags_it() -> None:
    merged = merge_provider_records(
        {"anilist": _record(nsfw=False), "kitsu": _record(nsfw=False)}
    )
    assert merged["nsfw"] is False


def test_nsfw_defaults_to_false_when_nobody_supplies_it() -> None:
    # The flag is never left empty: consumers always get a usable answer
    # rather than having to decide what an absent one means.
    assert merge_provider_records({"mal": _record()})["nsfw"] is False


def test_a_content_warning_outranks_every_provider() -> None:
    # MAL is trusted most, but calling a word a theme is not a denial that it
    # is adult content. The two mistakes are not equally bad.
    merged = merge_provider_records(
        {
            "mal": _record(themes=[{"name": "Nudity"}]),
            "anilist": _record(content_warnings=["Nudity"]),
        }
    )
    assert merged["content_warnings"] == ["Nudity"]
    assert merged["themes"] == []


def test_a_warning_does_not_also_sit_in_tags() -> None:
    # AniDB publishes 31 of AniList's adult-flagged words as ordinary tags.
    merged = merge_provider_records(
        {
            "anilist": _record(content_warnings=["Large Breasts"]),
            "anidb": _record(tags=["large breasts"]),
        }
    )
    assert merged["content_warnings"] == ["Large Breasts"]
    assert merged["tags"] == []


def test_words_nobody_flagged_are_untouched_by_the_warning_rule() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(genres=["Action"], themes=[{"name": "Military"}]),
            "anilist": _record(content_warnings=["Nudity"]),
        }
    )
    assert merged["genres"] == ["Action"]
    assert [t["name"] for t in merged["themes"]] == ["Military"]
    assert merged["content_warnings"] == ["Nudity"]
