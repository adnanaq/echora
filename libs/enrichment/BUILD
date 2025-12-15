"""
Enrichment - Anime data enrichment library.

Provides:
- API helpers for external anime data sources (AniList, Kitsu, Jikan, etc.)
- Character similarity matching (CCIP)
- Data crawlers and scrapers
- Multi-stage enrichment pipeline
"""

python_distribution(
    name="enrichment_dist",
    dependencies=[
        "//libs/enrichment/src/enrichment",
    ],
    provides=python_artifact(
        name="echora-enrichment",
        version="0.1.0",
        description="Anime data enrichment with API integrations and similarity matching",
    ),
    generate_setup=True,
)
