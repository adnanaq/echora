"""Configuration of the review agent, which suggests review answers from Jev's judgments."""

from pathlib import Path

from pydantic import Field, SecretStr

from .env_settings import EnvSettings


class AgentConfig(EnvSettings):
    """The TypeSafe key Jev is asked with, and the folder holding Jev's answers.

    Without a key, no pair is asked and the review page shows no suggestions.
    """

    typesafe_api_key: SecretStr | None = Field(
        default=None, description="TypeSafe API key Jev is asked with"
    )
    agent_data_dir: Path = Field(
        default=Path("data/seed_cleaning"),
        description=(
            "Seed cleaning's data folder, which holds the review file and "
            "Jev's answers about page pairs"
        ),
    )
