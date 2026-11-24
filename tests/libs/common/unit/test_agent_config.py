from pathlib import Path
from unittest.mock import patch

from common.config import AgentConfig, Settings


def test_agent_config_defaults_read_seed_cleaning_data() -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = AgentConfig()
    assert (config.typesafe_api_key, config.agent_data_dir) == (
        None,
        Path("data/seed_cleaning"),
    )


def test_agent_config_reads_environment_variables() -> None:
    with patch.dict(
        "os.environ",
        {
            "TYPESAFE_API_KEY": "typesafe-key",
            "AGENT_DATA_DIR": "/app/data/seed_cleaning",
        },
        clear=True,
    ):
        config = AgentConfig()
    assert config.typesafe_api_key is not None
    assert (
        config.typesafe_api_key.get_secret_value(),
        config.agent_data_dir,
    ) == ("typesafe-key", Path("/app/data/seed_cleaning"))


def test_settings_holds_agent_config() -> None:
    with patch.dict(
        "os.environ",
        {"ENVIRONMENT": "development", "TYPESAFE_API_KEY": "typesafe-key"},
        clear=True,
    ):
        settings = Settings()
    assert settings.agent.typesafe_api_key is not None
