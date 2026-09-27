#!/usr/bin/env python3
"""Summarize a py-spy profile of the vector service by thread and by area.

Reads the folded stacks written by ``diagnostics/profile_service.sh``
(``py-spy record --format raw``) and prints:

- how busy the event loop thread and the other threads (model and gRPC
  worker threads) were: samples divided by the samples a fully busy thread
  would produce (py-spy records nothing for an idle thread)
- where the event loop's time went, grouped into areas of the code
- the functions most often on top of each stack, per thread group

Usage: uv run python -m benchmarks.vector_service.diagnostics.summarize_profile PROFILE [--seconds 25] [--rate 200]
"""

import argparse
from pathlib import Path

from benchmarks.vector_service.toolkit.results import (
    print_profile_group,
    read_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", type=Path)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument(
        "--rate", type=int, default=200, help="py-spy samples per second"
    )
    arguments = parser.parse_args()
    counts = read_profile(arguments.profile)
    full_thread = arguments.seconds * arguments.rate
    print(
        f"event loop thread busy {sum(counts.loop_areas.values()) / full_thread:.0%};"
        f" other threads together {sum(counts.worker_areas.values()) / full_thread:.0%}"
        " of one thread"
    )
    print_profile_group("event loop thread", counts.loop_areas, counts.loop_leaves)
    print_profile_group("other threads", counts.worker_areas, counts.worker_leaves)


if __name__ == "__main__":
    main()
