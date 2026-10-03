from benchmarks.vector_service.toolkit.result_agreement import compare_top_results


def test_compare_top_results_identical_results_agree_fully():
    agreement = compare_top_results([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])

    assert agreement.identical == 2
    assert agreement.total == 2
    assert agreement.overlap == 1.0


def test_compare_top_results_same_ids_reordered_overlap_without_being_identical():
    agreement = compare_top_results([[1, 2, 3]], [[3, 2, 1]])

    assert agreement.identical == 0
    assert agreement.overlap == 1.0


def test_compare_top_results_overlap_counts_shared_ids_per_list():
    agreement = compare_top_results([[1, 2, 3, 4], [5, 6]], [[1, 2, 9, 8], [5, 6]])

    assert agreement.overlap == (2 + 2) / (4 + 2)


def test_compare_top_results_empty_results_agree():
    agreement = compare_top_results([[]], [[]])

    assert agreement.identical == 1
    assert agreement.overlap == 1.0
