"""
Common package - Shared models, configuration, and utilities.

This is a base package with no dependencies on other internal libs.
"""

python_distribution(
    name="common_dist",
    dependencies=[
        "//libs/common/src/common",
    ],
    provides=python_artifact(
        name="echora-common",
        version="0.1.0",
        description="Shared models, configuration, and utilities for Echora",
    ),
    # Generate sdist and wheel for potential distribution
    generate_setup=True,
)
