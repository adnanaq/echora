import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from enrichment.api_helpers.mal_enrichment_helper import MalEnrichmentHelper


def test_reexports_mal_enrichment_helper():
    from enrichment.api_helpers.mal_helper import MalEnrichmentHelper as Exported

    assert Exported is MalEnrichmentHelper


class TestMALHelperCli:
    @pytest.mark.asyncio
    async def test_main_anime_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "anime.json"

            helper = AsyncMock()
            helper.fetch_anime = AsyncMock(return_value={"mal_id": 1, "title": "X"})

            # Async context manager
            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.api_helpers.mal_helper.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                with patch(
                    "sys.argv",
                    ["mal_helper", "anime", "1", str(out)],
                ):
                    from enrichment.api_helpers.mal_helper import main

                    rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))["mal_id"] == 1

    @pytest.mark.asyncio
    async def test_main_episodes_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "episodes.json"

            helper = AsyncMock()
            helper.fetch_episodes = AsyncMock(
                return_value=[{"episode_number": 1}, {"episode_number": 2}]
            )

            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.api_helpers.mal_helper.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                with patch(
                    "sys.argv",
                    ["mal_helper", "episodes", "21", "2", str(out)],
                ):
                    from enrichment.api_helpers.mal_helper import main

                    rc = await main()

            assert rc == 0
            assert out.exists()
            assert len(json.loads(out.read_text(encoding="utf-8"))) == 2
            helper.fetch_episodes.assert_awaited_once_with(2)

    @pytest.mark.asyncio
    async def test_main_characters_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "characters.json"

            helper = AsyncMock()
            helper.fetch_characters_basic = AsyncMock(
                return_value=[{"character": {"mal_id": 10}, "role": "Main"}]
            )
            helper.fetch_characters_detailed = AsyncMock(
                return_value=[{"character_id": 10, "name": "A"}]
            )

            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.api_helpers.mal_helper.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                with patch(
                    "sys.argv",
                    ["mal_helper", "characters", "21", str(out)],
                ):
                    from enrichment.api_helpers.mal_helper import main

                    rc = await main()

            assert rc == 0
            assert out.exists()
            assert json.loads(out.read_text(encoding="utf-8"))[0]["character_id"] == 10
            helper.fetch_characters_basic.assert_awaited_once_with()
            helper.fetch_characters_detailed.assert_awaited_once()

