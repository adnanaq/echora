from datetime import UTC, datetime
from pathlib import Path

from match_suggestions.model_pair_judgment import PairJudgment
from match_suggestions.pair_judgment_store import JsonlPairJudgmentStore

PAIR = ("anilist:anime:2", "mal:anime:1")
OTHER_PAIR = ("kitsu:anime:3", "mal:anime:1")


def _judgment(
    page_key_pair: tuple[str, str], is_same_work: float, day: int = 1
) -> PairJudgment:
    return PairJudgment(
        first_page_key=page_key_pair[0],
        second_page_key=page_key_pair[1],
        question_version="version",
        request_fingerprint=f"fingerprint-{day}",
        is_same_franchise=0.9,
        is_same_work=is_same_work,
        is_first_work_part_of_second_work=0.1,
        is_second_work_part_of_first_work=0.1,
        judged_at=datetime(2026, 11, day, tzinfo=UTC),
    )


def test_find_judgments_reopened_store_returns_added_judgment(tmp_path: Path) -> None:
    store_path = tmp_path / "page_pair_judgments.jsonl"
    JsonlPairJudgmentStore(store_path).add_judgments([_judgment(PAIR, 0.9)])

    assert JsonlPairJudgmentStore(store_path).find_judgments([PAIR, OTHER_PAIR]) == {
        PAIR: _judgment(PAIR, 0.9)
    }


def test_find_judgments_pair_judged_twice_returns_latest(tmp_path: Path) -> None:
    store = JsonlPairJudgmentStore(tmp_path / "page_pair_judgments.jsonl")
    store.add_judgments([_judgment(PAIR, 0.9, day=1)])
    store.add_judgments([_judgment(PAIR, 0.2, day=2)])

    assert store.find_judgments([PAIR])[PAIR].is_same_work == 0.2


def test_find_judgments_missing_file_returns_nothing(tmp_path: Path) -> None:
    assert (
        JsonlPairJudgmentStore(tmp_path / "missing.jsonl").find_judgments([PAIR]) == {}
    )


def test_find_judgments_partial_last_line_left_out(tmp_path: Path) -> None:
    store_path = tmp_path / "page_pair_judgments.jsonl"
    JsonlPairJudgmentStore(store_path).add_judgments([_judgment(PAIR, 0.9)])
    with store_path.open("a", encoding="utf-8") as store_file:
        store_file.write('{"first_page_key": "kitsu')

    assert list(
        JsonlPairJudgmentStore(store_path).find_judgments([PAIR, OTHER_PAIR])
    ) == [PAIR]


def test_find_judgments_judgment_added_by_other_store_read_again(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "page_pair_judgments.jsonl"
    reading_store = JsonlPairJudgmentStore(store_path)
    assert reading_store.find_judgments([PAIR]) == {}

    JsonlPairJudgmentStore(store_path).add_judgments([_judgment(PAIR, 0.9)])

    assert reading_store.find_judgments([PAIR]) == {PAIR: _judgment(PAIR, 0.9)}
