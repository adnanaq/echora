import json
from pathlib import Path

from benchmarks.vector_service.toolkit.image_entities import (
    ImageEntity,
    collect_agent_entities,
    collect_database_entities,
    hold_out_queries,
)


def write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_characters_from_several_providers_merge_by_name(tmp_path: Path):
    folder = tmp_path / "One_agent5"
    folder.mkdir()
    write_lines(
        folder / "mal_characters.jsonl",
        [{"name": "Monkey D. Luffy", "images": ["https://a/1.jpg"]}],
    )
    write_lines(
        folder / "kitsu_characters.jsonl",
        [{"name": "monkey d luffy", "images": ["https://b/2.jpg"]}],
    )

    entities = collect_agent_entities([folder], is_cached=lambda url: True)

    assert entities == [
        ImageEntity(
            "One:character:monkeydluffy",
            "character",
            ("https://b/2.jpg", "https://a/1.jpg"),
        )
    ]


def test_agent_folders_of_one_anime_share_entities_and_drop_repeats(tmp_path: Path):
    for name in ("One_agent1", "One_refresh"):
        folder = tmp_path / name
        folder.mkdir()
        write_lines(
            folder / "mal_anime.jsonl",
            [{"images": {"covers": ["https://a/c.jpg"], "banners": []}}],
        )
    anime_folder = tmp_path / "One_agent2"
    anime_folder.mkdir()
    write_lines(
        anime_folder / "kitsu_anime.jsonl",
        [{"images": {"posters": ["https://b/p.jpg"]}}],
    )

    entities = collect_agent_entities(
        sorted(tmp_path.iterdir()), is_cached=lambda url: True
    )

    assert entities == [
        ImageEntity("One:anime", "anime", ("https://a/c.jpg", "https://b/p.jpg"))
    ]


def test_episodes_are_keyed_by_number_and_uncached_images_dropped(tmp_path: Path):
    folder = tmp_path / "Death_agent6"
    folder.mkdir()
    write_lines(
        folder / "kitsu_episodes.jsonl",
        [
            {"episode_number": 1, "images": ["https://k/e1.jpg"]},
            {"episode_number": 2, "images": ["https://k/missing.jpg"]},
        ],
    )

    entities = collect_agent_entities(
        [folder], is_cached=lambda url: "missing" not in url
    )

    assert entities == [
        ImageEntity("Death:episode:1", "episode", ("https://k/e1.jpg",))
    ]


def test_database_file_gives_anime_and_their_characters(tmp_path: Path):
    path = tmp_path / "anime_database.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "anime": {
                            "title": "Nvade Show",
                            "images": {
                                "covers": ["https://a/1.jpg", "https://b/2.jpg"]
                            },
                        },
                        "characters": [{"name": "Rui", "images": ["https://c/3.jpg"]}],
                    }
                ]
            }
        )
    )

    entities = collect_database_entities(path, is_cached=lambda url: True)

    assert entities == [
        ImageEntity(
            "db:nvadeshow:anime", "anime", ("https://a/1.jpg", "https://b/2.jpg")
        ),
        ImageEntity("db:nvadeshow:character:rui", "character", ("https://c/3.jpg",)),
    ]


def test_one_image_per_multi_image_entity_becomes_the_query():
    entities = [
        ImageEntity("a", "anime", ("a1", "a2", "a3")),
        ImageEntity("b", "character", ("b1",)),
    ]

    stored, queries = hold_out_queries(entities, seed=4)

    assert [key for key, _ in queries] == ["a"]
    held_out = queries[0][1]
    assert held_out in ("a1", "a2", "a3")
    assert stored == {
        "a": tuple(url for url in ("a1", "a2", "a3") if url != held_out),
        "b": ("b1",),
    }


def test_same_seed_holds_out_the_same_images():
    entities = [
        ImageEntity(str(n), "character", (f"{n}x", f"{n}y", f"{n}z")) for n in range(20)
    ]

    assert hold_out_queries(entities, seed=7) == hold_out_queries(entities, seed=7)
