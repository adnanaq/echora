"""Unit tests for VectorDocument with multivector support."""

import pytest
from vector_db_interface import VectorDocument


def test_vector_document_single_vectors():
    """Test VectorDocument with single vectors (backward compatible)."""
    doc = VectorDocument(
        id="test-id",
        vectors={"text_vector": [0.1, 0.2, 0.3]},
        payload={"type": "anime"},
    )
    assert doc.id == "test-id"
    assert doc.vectors["text_vector"] == [0.1, 0.2, 0.3]


def test_vector_document_multivector():
    """Test VectorDocument with multivector (list of lists)."""
    doc = VectorDocument(
        id="test-id",
        vectors={
            "text_vector": [0.1, 0.2, 0.3],
            "image_vector": [
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        },
        payload={"type": "anime"},
    )
    assert doc.id == "test-id"
    assert len(doc.vectors["image_vector"]) == 2
    assert doc.vectors["image_vector"][0] == [0.4, 0.5, 0.6]
