"""Stubs for heavy transitive imports that are unavailable outside Pants.

``enrichment.pipeline`` → ``api_fetcher`` → ``anilist_helper`` →
``deduplication`` → ``vector_processing`` (ML models, GPU deps).

Stubbing at sys.modules level lets pytest collect and run id_extractor
tests without the full ML stack installed.
"""

import sys
from unittest.mock import MagicMock

for _mod in (
    "vector_processing",
    "vector_processing.embedding_models",
    "vector_processing.embedding_models.text",
    "vector_processing.embedding_models.text.base",
    "vector_db_interface",
    "vector_db_interface.base",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
