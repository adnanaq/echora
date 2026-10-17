"""
Work Matching - decide which provider pages belong to the same work.

Holds the work entry, provider page keys and details, and the known-works
lookup that seed cleaning, seed link filling and discovery compare against.
Imports nothing from the enrichment pipeline.
"""

python_distribution(
    name="work_matching_dist",
    dependencies=[
        "//libs/work_matching/src/work_matching",
    ],
    provides=python_artifact(
        name="echora-work-matching",
        version="0.1.0",
        description="Work entries, provider page keys and a known-works lookup",
    ),
    generate_setup=True,
)
