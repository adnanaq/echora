from typing import Any

import numpy

def cosine_similarity(
    X: numpy.ndarray[Any] | list[list[float]],
    Y: numpy.ndarray[Any] | list[list[float]] | None = None,
) -> numpy.ndarray[Any]: ...
