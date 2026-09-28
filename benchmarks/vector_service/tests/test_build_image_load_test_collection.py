import numpy as np

from benchmarks.vector_service.test_data.build_image_load_test_collection import (
    EntityImages,
    ImagePoint,
    image_points,
    to_qdrant_point,
)


def test_points_carry_their_entity_type_and_one_to_max_images():
    kinds = [EntityImages("anime", 5, 6), EntityImages("character", 7, 2)]
    points = list(image_points(kinds, dimensions=8, seed=1, first_id=100))

    assert [point.point_id for point in points] == list(range(100, 112))
    assert [point.entity_type for point in points] == ["anime"] * 5 + ["character"] * 7
    assert all(1 <= len(point.images) <= 6 for point in points[:5])
    assert all(1 <= len(point.images) <= 2 for point in points[5:])


def test_image_vectors_are_unit_length():
    points = list(image_points([EntityImages("anime", 3, 4)], dimensions=16, seed=2))

    for point in points:
        norms = np.linalg.norm(np.array(point.images), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_same_seed_gives_the_same_points():
    kinds = [EntityImages("anime", 4, 6)]

    assert list(image_points(kinds, 8, seed=3)) == list(image_points(kinds, 8, seed=3))


def test_qdrant_point_holds_the_images_under_the_vector_name():
    point = to_qdrant_point(ImagePoint(7, "anime", [[1.0, 0.0]]), "image_vector")

    assert point.id == 7
    assert point.vector == {"image_vector": [[1.0, 0.0]]}
    assert point.payload == {"entity_type": "anime"}
