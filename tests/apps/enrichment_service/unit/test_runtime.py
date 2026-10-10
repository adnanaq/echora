from unittest.mock import patch

from common.config import CacheConfig, RedisConfig, ServiceConfig, Settings
from enrichment_service.runtime import build_runtime
from http_cache import get_cache


async def test_build_runtime_configures_cache_and_keeps_enrichment_config() -> None:
    with patch.dict("os.environ", {"ENVIRONMENT": "development"}):
        settings = Settings(
            service=ServiceConfig(
                enrichment_default_file_path="db.json", enrichment_output_dir="out"
            ),
            cache=CacheConfig(cache_enabled=True, ttl_kitsu=60),
            redis=RedisConfig(redis_url="redis://localhost:6379/15"),
        )

    runtime = await build_runtime(settings)

    assert runtime.default_file_path == "db.json"
    assert runtime.output_dir == "out"
    assert runtime.enrichment is settings.enrichment
    assert runtime.cache is get_cache()
    assert runtime.cache.enabled is True
    assert runtime.cache.cache.ttl_for("kitsu") == 60
