"""
Enrichment - Anime data enrichment library.

Provides:
- API helpers for external anime data sources (AniList, Kitsu, MAL, etc.)
- Character similarity matching (CCIP)
- Data crawlers and scrapers
- Multi-stage enrichment pipeline
"""

python_distribution(
    name="enrichment_dist",
    dependencies=[
        "//libs/enrichment/src/enrichment",
        # Transitively exclude type stubs (from both direct and transitive deps via vector_processing)
        "!!stubs/crawl4ai:crawl4ai",
        "!!stubs/fastembed:fastembed",  # Transitive via vector_processing
        "!!stubs/imgutils:imgutils",
        "!!stubs/numpy:numpy",
        "!!stubs/sklearn:sklearn",
        "!!stubs/torch:torch",  # Transitive via vector_processing
    ],
    provides=python_artifact(
        name="echora-enrichment",
        version="0.1.0",
        description="Anime data enrichment with API integrations and similarity matching",
    ),
    generate_setup=True,
)
