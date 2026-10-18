"""
Seed Cleaning - find and fetch the seed entries that need checking.

Built on work_matching and the enrichment helpers. Deleted once discovery
replaces the seed.
"""

python_distribution(
    name="seed_cleaning_dist",
    dependencies=[
        "//libs/seed_cleaning/src/seed_cleaning",
    ],
    provides=python_artifact(
        name="echora-seed-cleaning",
        version="0.1.0",
        description="Seed entries to check and the provider page fetcher",
    ),
    generate_setup=True,
)
