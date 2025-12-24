from typing import Any, Generator, List

class TextEmbedding:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        **kwargs: Any
    ) -> None: ...
    
    def embed(self, texts: List[str]) -> Generator[Any, None, None]: ...
