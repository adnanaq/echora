#!/usr/bin/env python3
"""
Multi-Agent Concurrency Integration Tests

Tests concurrent execution of multiple enrichment agents to verify:
1. Concurrent agent directory creation without collisions
2. Concurrent Redis cache access without race conditions
3. Agent ID assignment under concurrent load (gap-filling logic)
4. Shared cache behavior with multiple agents

These are integration tests that simulate real-world concurrent enrichment scenarios.

NOTE: Tests that require live AniList API and Redis are skipped by default.
Set ENABLE_LIVE_CONCURRENCY_TESTS=1 to run live tests with real API/Redis.
"""

import asyncio
import os
import shutil
import tempfile
import time

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Skip live API/Redis tests unless explicitly enabled
ENABLE_LIVE_TESTS = os.getenv("ENABLE_LIVE_CONCURRENCY_TESTS")
skip_live_tests = pytest.mark.skipif(
    not ENABLE_LIVE_TESTS,
    reason="Live API/Redis tests disabled. Set ENABLE_LIVE_CONCURRENCY_TESTS=1 to enable.",
)


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for multi-agent testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_multi_agent_")
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config(temp_test_dir):
    """Create mock EnrichmentConfig for testing."""
    from enrichment.programmatic.config import EnrichmentConfig

    config = EnrichmentConfig(temp_dir=temp_test_dir)
    return config


class TestConcurrentAgentDirectoryCreation:
    """Test concurrent agent directory creation without collisions."""

    @pytest.mark.asyncio
    async def test_concurrent_agents_create_unique_directories(self, mock_config):
        """
        Test that multiple agents running concurrently create unique directories
        without ID collisions.

        Expected behavior:
        - 5 concurrent agents for same anime create directories with IDs 1-5
        - No ID collisions (each agent gets unique ID)
        - All directories exist after concurrent creation
        """
        from enrichment.programmatic.enrichment_pipeline import (
            ProgrammaticEnrichmentPipeline,
        )

        anime_title = "TestAnime"
        num_agents = 5
        created_dirs = []

        async def create_agent_dir(agent_num: int) -> str:
            """Simulate concurrent agent directory creation."""
            pipeline = ProgrammaticEnrichmentPipeline(config=mock_config)
            # Create temp directory (triggers agent ID assignment)
            temp_dir = pipeline._create_temp_dir(anime_title)
            return temp_dir

        # Run 5 agents concurrently
        tasks = [create_agent_dir(i) for i in range(num_agents)]
        created_dirs = await asyncio.gather(*tasks)

        # Verify all directories created
        assert len(created_dirs) == num_agents, (
            f"Expected {num_agents} dirs, got {len(created_dirs)}"
        )

        # Verify all directories are unique (no collisions)
        unique_dirs = set(created_dirs)
        assert len(unique_dirs) == num_agents, (
            f"Directory collision detected! Expected {num_agents} unique dirs, got {len(unique_dirs)}. Dirs: {created_dirs}"
        )

        # Verify all directories actually exist
        for dir_path in created_dirs:
            assert os.path.exists(dir_path), f"Directory not created: {dir_path}"

        # Verify agent IDs are sequential (1-5)
        agent_ids = []
        for dir_path in created_dirs:
            dir_name = os.path.basename(dir_path)
            # Extract agent ID from format: TestAnime_agent<N>
            agent_id_str = dir_name.split("_agent")[1]
            agent_ids.append(int(agent_id_str))

        agent_ids.sort()
        assert agent_ids == list(range(1, num_agents + 1)), (
            f"Expected agent IDs 1-{num_agents}, got {agent_ids}"
        )


class TestConcurrentRedisAccess:
    """Test concurrent Redis cache access from multiple agents."""

    @skip_live_tests
    @pytest.mark.asyncio
    async def test_concurrent_agents_share_redis_cache(self, mock_config):
        """
        Test that multiple agents can concurrently access Redis cache
        without race conditions or data corruption.

        Expected behavior:
        - Multiple agents fetch same API data concurrently
        - First agent caches response, others hit cache
        - All agents receive same correct data
        - No cache corruption or race conditions

        NOTE: This is a real integration test requiring:
        - Redis server running
        - Internet connection for API calls
        - AniList API access

        Set ENABLE_LIVE_CONCURRENCY_TESTS=1 to run this test.
        """
        from enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

        num_agents = 3
        anilist_id = 1535  # Death Note (much smaller anime)

        # Track cache hits
        cache_hits = []

        async def agent_fetch_data(agent_id: int) -> dict:
            """Simulate agent fetching data with Redis cache."""
            helper = AniListEnrichmentHelper()

            # Use _make_request directly to get _from_cache metadata
            query = helper._build_query_by_anilist_id()
            variables = {"id": anilist_id}
            data = await helper._make_request(query, variables)

            # Track if this was from cache
            is_cached = data.get("_from_cache", False) if data else False
            cache_hits.append((agent_id, is_cached))

            return data

        # Run 3 agents concurrently
        tasks = [agent_fetch_data(i) for i in range(num_agents)]
        results = await asyncio.gather(*tasks)

        # Verify all agents got data
        assert all(results), "All agents should receive data"

        # Verify all agents got same data (no corruption)
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            # Compare Media ID (skip _from_cache metadata)
            first_media_id = first_result.get("Media", {}).get("id")
            result_media_id = result.get("Media", {}).get("id")
            assert result_media_id == first_media_id, (
                f"Agent {i} got different data! Cache corruption detected. "
                f"Expected ID {first_media_id}, got {result_media_id}"
            )

        # With true concurrent execution, all agents may start before any completes,
        # resulting in 0 cache hits (all fetch from API). This is valid behavior.
        # The important validation is data consistency (verified above).
        cached_count = sum(1 for _, is_cached in cache_hits if is_cached)

        # Log cache behavior for debugging (but don't assert on it)
        print(f"\nCache behavior: {cached_count}/{num_agents} cache hits")
        print(f"Cache hits details: {cache_hits}")

    @pytest.mark.asyncio
    async def test_concurrent_http_cache_manager_client_creation(self):
        """
        Test concurrent get_aiohttp_session() calls on same HTTPCacheManager instance.

        This test verifies that the check-then-act pattern in _get_or_create_redis_client()
        does not create race conditions when multiple coroutines concurrently request sessions.

        Expected behavior:
        - Only ONE Redis client is created despite concurrent calls
        - All coroutines receive the same client instance
        - No duplicate client creation

        This validates that Python's GIL and synchronous execution of _get_or_create_redis_client()
        prevents race conditions without needing explicit locking.
        """
        from unittest.mock import MagicMock, patch

        from http_cache.config import CacheConfig
        from http_cache.manager import HTTPCacheManager

        config = CacheConfig(
            enabled=True,
            storage_type="redis",
            redis_url="redis://localhost:6379/0",
        )

        # Track Redis client creation calls
        creation_count = [0]
        created_clients = []

        async def async_close():
            """Mock async close."""
            pass

        def mock_from_url(*args, **kwargs):
            """Mock that tracks creation calls."""
            creation_count[0] += 1
            client = MagicMock()
            client.close = async_close
            created_clients.append(client)
            return client

        with patch("redis.asyncio.Redis.from_url", side_effect=mock_from_url):
            with patch("http_cache.async_redis_storage.AsyncRedisStorage"):
                with patch("http_cache.aiohttp_adapter.CachedAiohttpSession"):
                    manager = HTTPCacheManager(config)

                    # Launch 10 concurrent get_aiohttp_session() calls
                    async def get_session(task_id: int):
                        session = manager.get_aiohttp_session(f"service_{task_id}")
                        client_id = id(manager._async_redis_client)
                        return (task_id, session, client_id)

                    tasks = [get_session(i) for i in range(10)]
                    results = await asyncio.gather(*tasks)

                    # Verify only ONE Redis client was created
                    assert creation_count[0] == 1, (
                        f"Race condition detected! Expected 1 Redis client creation, "
                        f"but got {creation_count[0]} creations. This indicates multiple "
                        f"concurrent calls created duplicate clients."
                    )

                    # Verify all tasks got the same client instance
                    client_ids = [client_id for _, _, client_id in results]
                    unique_clients = set(client_ids)
                    assert len(unique_clients) == 1, (
                        f"Client instance mismatch! Expected all tasks to get same client, "
                        f"but got {len(unique_clients)} different client IDs: {unique_clients}"
                    )

        # The test validates that concurrent access doesn't cause corruption
        # Cache hit count is timing-dependent and not a valid assertion


class TestAgentIDRaceConditions:
    """Test agent ID assignment under concurrent load."""

    @pytest.mark.asyncio
    async def test_gap_filling_under_concurrent_load(self, mock_config):
        """
        Test that gap-filling logic works correctly when multiple agents
        create directories concurrently.

        Scenario:
        1. Create agents 1, 2, 4 (leaving gap at 3)
        2. Launch 3 concurrent agents
        3. Verify one agent fills gap (gets ID 3)
        4. Verify others get sequential IDs (5, 6)

        Expected: No duplicate IDs, gap filled correctly
        """
        from enrichment.programmatic.enrichment_pipeline import (
            ProgrammaticEnrichmentPipeline,
        )

        # Pre-create directories with gap
        anime_title = "GapTest"
        for agent_id in [1, 2, 4]:
            dir_path = os.path.join(
                mock_config.temp_dir, f"{anime_title}_agent{agent_id}"
            )
            os.makedirs(dir_path, exist_ok=True)

        # Now launch 3 concurrent agents
        num_concurrent = 3
        created_dirs = []

        async def create_agent_with_gap() -> str:
            """Create agent that should fill gap or get next ID."""
            pipeline = ProgrammaticEnrichmentPipeline(config=mock_config)
            temp_dir = pipeline._create_temp_dir(anime_title)
            return temp_dir

        tasks = [create_agent_with_gap() for _ in range(num_concurrent)]
        created_dirs = await asyncio.gather(*tasks)

        # Extract agent IDs from created directories
        new_agent_ids = []
        for dir_path in created_dirs:
            dir_name = os.path.basename(dir_path)
            agent_id_str = dir_name.split("_agent")[1]
            new_agent_ids.append(int(agent_id_str))

        new_agent_ids.sort()

        # Verify gap was filled and sequential IDs assigned
        # Expected: [3, 5, 6] (fills gap 3, then continues from 4)
        expected_ids = [3, 5, 6]
        assert new_agent_ids == expected_ids, (
            f"Gap-filling failed under concurrent load. "
            f"Expected {expected_ids}, got {new_agent_ids}"
        )

        # Verify no duplicate IDs across all agents
        all_agent_ids = [1, 2, 4] + new_agent_ids
        assert len(all_agent_ids) == len(set(all_agent_ids)), (
            f"Duplicate agent IDs detected: {all_agent_ids}"
        )


class TestConcurrentPipelineExecution:
    """Test full enrichment pipeline with concurrent agents."""

    @skip_live_tests
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution_stress(self, mock_config):
        """
        Stress test: Run multiple full enrichment pipelines concurrently.

        This simulates real-world scenario of multiple anime being enriched
        simultaneously with shared Redis cache.

        Expected behavior:
        - All pipelines complete successfully
        - No race conditions in cache
        - Agent directories properly isolated
        - Cache hits occur for shared data

        NOTE: Requires live AniList API and Redis.
        Set ENABLE_LIVE_CONCURRENCY_TESTS=1 to run this test.
        """
        from enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

        # Use small anime for stress test (avoid huge episode/character lists)
        anime_configs = [
            {"id": 1535, "title": "DeathNote"},  # 37 episodes
            {"id": 5114, "title": "FullmetalAlchemistBrotherhood"},  # 64 episodes
            {"id": 16498, "title": "AttackOnTitan"},  # 25 episodes (S1)
        ]

        results = []

        async def run_enrichment_pipeline(anime_config: dict) -> dict:
            """Run full enrichment for one anime."""
            helper = AniListEnrichmentHelper()

            # Fetch from AniList concurrently
            start_time = time.time()

            # Use only fast service for stress test
            anilist_data = await helper.fetch_anime_by_anilist_id(anime_config["id"])

            elapsed = time.time() - start_time

            return {
                "anime_id": anime_config["id"],
                "title": anime_config["title"],
                "success": anilist_data is not None,
                "elapsed": elapsed,
                "from_cache": (
                    anilist_data.get("_from_cache", False) if anilist_data else False
                ),
            }

        # Run 3 pipelines concurrently
        tasks = [run_enrichment_pipeline(config) for config in anime_configs]
        results = await asyncio.gather(*tasks)

        # Verify all pipelines completed
        assert len(results) == len(anime_configs), "Not all pipelines completed"

        # Verify all successful
        success_count = sum(1 for r in results if r["success"])
        assert success_count == len(anime_configs), (
            f"Only {success_count}/{len(anime_configs)} pipelines succeeded. "
            f"Results: {results}"
        )

        # Performance check: concurrent execution should be faster than sequential
        # (This is optional - just validating concurrency works)
        max_elapsed = max(r["elapsed"] for r in results)
        total_elapsed = sum(r["elapsed"] for r in results)

        # If truly concurrent, max_elapsed << total_elapsed
        # We don't assert this strictly as it depends on network/cache state
        print(
            f"\nConcurrency efficiency: {max_elapsed:.2f}s concurrent vs {total_elapsed:.2f}s sequential potential"
        )
