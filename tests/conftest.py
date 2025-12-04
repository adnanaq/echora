"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

import pytest
from common.config.settings import get_settings


@pytest.fixture(scope="session")
def settings():
    """Get test settings with test collection name."""
    settings = get_settings()
    # Override to use test collection for ALL tests
    settings.qdrant_collection_name = "anime_database_test"
    return settings
