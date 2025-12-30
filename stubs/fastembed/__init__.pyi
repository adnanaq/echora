from collections.abc import Generator
from typing import Any

class TextEmbedding:
    def __init__(
        self, model_name: str, cache_dir: str | None = None, **kwargs: Any
    ) -> None: ...
    def embed(self, texts: list[str]) -> Generator[Any, None, None]: ...
