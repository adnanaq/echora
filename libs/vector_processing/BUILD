"""
Vector Processing - Multi-vector embedding generation and processing.

Provides:
- Text embeddings (BGE-M3, Sentence Transformers)
- Vision embeddings (OpenCLIP)
- Multi-vector embedding management
- Anime field mapping for vector generation
"""

python_distribution(
    name="vector_processing_dist",
    dependencies=[
        "//libs/vector_processing/src/vector_processing",
    ],
    provides=python_artifact(
        name="echora-vector-processing",
        version="0.1.0",
        description="Multi-vector embedding generation and processing for anime content",
    ),
    generate_setup=True,
)
