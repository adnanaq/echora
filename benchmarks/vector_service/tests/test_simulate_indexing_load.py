from benchmarks.vector_service.diagnostics.simulate_indexing_load import document_texts

QUERIES = ["pirate crew", "mecha war in space", "a quiet story about a small town cafe"]


def test_documents_reach_the_requested_length():
    documents = document_texts(QUERIES, 5, words=60, seed=1)

    assert len(documents) == 5
    assert all(len(document.split()) >= 60 for document in documents)


def test_same_seed_gives_the_same_documents():
    assert document_texts(QUERIES, 3, 40, seed=2) == document_texts(
        QUERIES, 3, 40, seed=2
    )
