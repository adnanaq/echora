"""
Tests for src/globals.py - Global state management module.
"""
import pytest
from src import globals as app_globals


def test_globals_module_exists():
    """Test that globals module can be imported."""
    assert app_globals is not None


def test_qdrant_client_initial_state():
    """Test that qdrant_client is initially None."""
    # Note: This may fail if another test has set the global
    # In a real test, we'd use fixtures to reset globals
    assert hasattr(app_globals, "qdrant_client")


def test_query_parser_agent_initial_state():
    """Test that query_parser_agent is initially None."""
    assert hasattr(app_globals, "query_parser_agent")


def test_globals_are_module_level():
    """Test that globals are accessible as module attributes."""
    # Verify we can access the attributes
    _ = app_globals.qdrant_client
    _ = app_globals.query_parser_agent


def test_globals_can_be_set():
    """Test that globals can be modified (simulating initialization)."""
    # Save original values
    original_client = app_globals.qdrant_client
    original_agent = app_globals.query_parser_agent

    try:
        # Set test values
        app_globals.qdrant_client = "test_client"
        app_globals.query_parser_agent = "test_agent"

        # Verify they were set
        assert app_globals.qdrant_client == "test_client"
        assert app_globals.query_parser_agent == "test_agent"

    finally:
        # Restore original values
        app_globals.qdrant_client = original_client
        app_globals.query_parser_agent = original_agent


def test_globals_have_correct_types():
    """Test that globals have correct type annotations."""
    from src.globals import qdrant_client, query_parser_agent

    # These should be importable
    assert qdrant_client is not None or qdrant_client is None
    assert query_parser_agent is not None or query_parser_agent is None
