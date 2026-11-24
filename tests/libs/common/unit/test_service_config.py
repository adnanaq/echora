from unittest.mock import patch

from common.config import ServiceConfig


def test_service_config_agent_service_defaults_listen_on_port_8005() -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = ServiceConfig()
    assert (config.agent_service_host, config.agent_service_port) == ("0.0.0.0", 8005)  # noqa: S104


def test_service_config_reads_agent_service_port() -> None:
    with patch.dict("os.environ", {"AGENT_SERVICE_PORT": "9005"}, clear=True):
        assert ServiceConfig().agent_service_port == 9005
