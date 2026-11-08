import pytest
from review_page.review_case_loader import ReviewCaseLoader, UnreadableReviewFileError
from work_matching import replace_file_lines


@pytest.fixture
def review_file_path(tmp_path, review_case_lines):
    path = tmp_path / "review.jsonl"
    replace_file_lines(path, review_case_lines)
    return path


def test_load_cases_unchanged_file_reuses_converted_cases(review_file_path) -> None:
    reader = ReviewCaseLoader(review_file_path)
    first_cases = reader.load_cases()
    assert reader.load_cases() is first_cases


def test_load_cases_replaced_file_reads_new_cases(
    review_file_path, review_case_lines
) -> None:
    reader = ReviewCaseLoader(review_file_path)
    first_cases = reader.load_cases()
    replace_file_lines(review_file_path, review_case_lines[:1])
    second_cases = reader.load_cases()
    assert (len(first_cases.cases), len(second_cases.cases)) == (
        len(review_case_lines),
        1,
    )


def test_load_cases_one_changed_line_parses_only_that_line(
    review_file_path, review_case_lines
) -> None:
    reader = ReviewCaseLoader(review_file_path)
    first_cases = reader.load_cases()
    answered_case = first_cases.cases[1].model_copy(
        update={"answer": {"anidb:anime:19188": "work 1"}}
    )
    replace_file_lines(
        review_file_path,
        [
            review_case_lines[0],
            answered_case.model_dump_json() + "\n",
            *review_case_lines[2:],
        ],
    )
    second_cases = reader.load_cases()
    unchanged_positions = [
        position for position in range(len(review_case_lines)) if position != 1
    ]
    assert (
        all(
            second_cases.cases[position] is first_cases.cases[position]
            and second_cases.case_messages[position]
            is first_cases.case_messages[position]
            for position in unchanged_positions
        ),
        second_cases.cases[1] is first_cases.cases[1],
        second_cases.cases[1].answer,
    ) == (True, False, {"anidb:anime:19188": "work 1"})


def test_load_cases_messages_match_cases(review_file_path) -> None:
    review_file_cases = ReviewCaseLoader(review_file_path).load_cases()
    assert [message.case_name for message in review_file_cases.case_messages] == [
        case.case_name for case in review_file_cases.cases
    ]


def test_load_cases_missing_file_gives_no_cases(tmp_path) -> None:
    review_file_cases = ReviewCaseLoader(tmp_path / "review.jsonl").load_cases()
    assert (review_file_cases.cases, review_file_cases.case_messages) == ((), ())


def test_load_cases_broken_line_raises_error_naming_line(
    review_file_path, review_case_lines
) -> None:
    replace_file_lines(review_file_path, [review_case_lines[0], "{not json}\n"])
    with pytest.raises(UnreadableReviewFileError, match="Line 2 of review.jsonl"):
        ReviewCaseLoader(review_file_path).load_cases()
