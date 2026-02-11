"""Executable module for starting the internal agent gRPC service."""

from __future__ import annotations

import os

# Common settings require ENVIRONMENT; default for local/dev.
os.environ.setdefault("ENVIRONMENT", "development")

from agent_service.server import main  # noqa: E402


if __name__ == "__main__":
    main()
