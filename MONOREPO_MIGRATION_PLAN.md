# Monorepo Migration Plan

## Guiding Principles & Engineer's Overview

This document outlines the technical strategy for transitioning our current single-package codebase into a scalable, modular monorepo, as defined in `docs/implementation_plan.md`. This is a foundational step that unlocks the larger cloud architecture. The primary goals are to improve code reusability, clarify ownership, isolate dependencies for leaner services, and enable parallel development.

The core philosophy is to "sharpen the saw"—investing time now in a robust structure to dramatically increase future development velocity and code quality. We will be disciplined and iterative, ensuring each step is verified before proceeding to the next.

---

## Phase 1: Scaffolding & Setup

**Goal:** Create the complete, empty directory structure for all new packages.

- [ ] Create the root `apps` and `libs` directories.
- [ ] Create application package directories for services that will be deployed:
  - [ ] `apps/agent`
  - [ ] `apps/lambdas/process_stage1_metadata`
  - [ ] `apps/lambdas/process_stage2_episodes`
  - [ ] `apps/lambdas/process_stage3_relationships`
  - [ ] `apps/lambdas/process_stage4_statistics`
  - [ ] `apps/lambdas/process_stage5_characters`
- [ ] Create library package directories for shared, reusable code:
  - [x] ~~`libs/common`~~
  - [x] ~~`libs/vector_db_interface`~~
  - [x] ~~`libs/qdrant_client`~~
  - [x] ~~`libs/vector_processing`~~
  - [ ] `libs/mongo_client`
  - [ ] `libs/cache_manager`
  - [ ] `libs/enrichment`
  - [ ] `libs/validation`
  - [ ] `libs/protos`
- [ ] Create a `src/<package_name>` structure within each new app and lib directory.
- [ ] Create a basic `README.md` in each new package directory.

## Phase 2: Code Migration

**Goal:** Move all existing code from the old `src/` directory into its new home in the appropriate `apps/` or `libs/` package.

- [ ] **Migration Strategy:** Follow a "bottom-up" approach. Migrate libraries with the fewest internal dependencies first (e.g., `libs/common`, `libs/vector_db_interface`), followed by libraries that depend on them, and finally the applications in `apps/`.
- [ ] **Script Refactoring:** When moving scripts to `apps/lambdas`, refactor them to have a clear `handler(event, context)` entry point suitable for serverless execution.
- [ ] Move `src/config` -> `libs/common/src/common/config`
- [x] ~~Move `src/models` -> `libs/common/src/common/models`~~
- [ ] Move `src/vector/client` -> `libs/qdrant_client/src/qdrant_client`
- [ ] Move remaining contents of `src/vector` -> `libs/vector_processing/src/vector_processing`
- [ ] Move `src/cache_manager` -> `libs/cache_manager/src/cache_manager`
- [ ] Move `src/enrichment` -> `libs/enrichment/src/enrichment`
- [ ] Move `src/validation` -> `libs/validation/src/validation`
- [ ] Move `src/main.py` -> `apps/agent/src/agent/main.py`
- [ ] Move `src/api` -> `apps/agent/src/agent/api`
- [ ] Move `src/server.py` -> `apps/agent/src/agent/server.py`
- [ ] Move `src/admin_service.py` -> `apps/agent/src/agent/admin_service.py`
- [ ] Move `src/agent_service.py` -> `apps/agent/src/agent/agent_service.py`
- [ ] Move `src/globals.py` -> `apps/agent/src/agent/globals.py`
- [ ] Move `src/poc` out of `src` to a top-level `poc/` directory.
- [ ] Move `scripts/process_stage1_metadata.py` -> `apps/lambdas/process_stage1_metadata/src/main.py`
- [ ] Move `scripts/process_stage2_episodes.py` -> `apps/lambdas/process_stage2_episodes/src/main.py`
- [ ] Move `scripts/process_stage3_relationships.py` -> `apps/lambdas/process_stage3_relationships/src/main.py`
- [ ] Move `scripts/process_stage4_statistics.py` -> `apps/lambdas/process_stage4_statistics/src/main.py`
- [ ] Move `scripts/process_stage5_characters.py` -> `apps/lambdas/process_stage5_characters/src/main.py`
- [ ] Handle `protos` directory (Note: This directory exists on another branch, this step is a placeholder for when it's integrated).
- [ ] Remove the now-empty `src/` directory.
- [ ] Review remaining files in `/scripts` to determine if they should be moved or kept as developer utilities.

## Phase 3: Build System Configuration

**Goal:** Configure the monorepo with an advanced build system (e.g., Pants) to manage dependencies, builds, and tests efficiently.

- [ ] **Initialize Build System:** Set up the chosen build system (e.g., Pants) in the repository root. This will create the necessary configuration files (e.g., `pants.toml`).
- [ ] **Configure Tooling:** Configure the build system to manage tools like `ruff`, `mypy`, and `pytest`.
- [ ] **Define Targets:** For each package in `apps/` and `libs/`, create `BUILD` files.
  - [ ] In each `BUILD` file, define the necessary targets (e.g., `python_sources`, `python_tests`, `pex_binary` for applications).
  - [ ] Declare dependencies for each target, specifying both third-party requirements and internal dependencies on other packages (e.g., `//libs/common`).
- [ ] **Dependency Strategy:** The root `pyproject.toml` will now primarily be used for global tool configuration. All production dependencies will be managed by the build system through the `BUILD` files, ensuring a granular and explicit dependency graph.

## Phase 4: Refactoring & Verification

**Goal:** Execute the key architectural improvements enabled by the new structure and verify the entire system works correctly using the build tool.

- [ ] **Iterative Verification:** Do not wait until the end of the migration to verify. After each significant code move or refactoring, run the full verification suite using the build tool's commands (e.g., `pants lint ::`, `pants check ::`, `pants test ::`) to catch issues early.
- [ ] **Architectural Refactoring (Vector Database):** Introduce a generic interface for vector database clients to enable pluggability.
  - [ ] Create a new `libs/vector_db_interface` library containing an abstract `VectorDBClient` base class.
  - [ ] Refactor `QdrantClient` in `libs/qdrant_client` to implement the `VectorDBClient` interface.
  - [ ] Decouple vector generation from the `QdrantClient`, making it a pure database client that accepts pre-generated `PointStruct` objects.
  - [ ] Update application code to depend on the `VectorDBClient` interface rather than the concrete `QdrantClient` implementation, using a factory to select the provider from settings.
  - [ ] <details>
                                    <summary>Example Code</summary>

        ```python
        # In libs/vector_db_interface/src/vector_db_interface/base.py
        from abc import ABC, abstractmethod
        class VectorDBClient(ABC):
            @abstractmethod
            async def upsert_points(self, collection_name: str, points: list) -> bool:
                pass

        # In libs/qdrant_client/src/qdrant_client/client.py
        from vector_db_interface.base import VectorDBClient
        class QdrantClient(VectorDBClient):
            async def upsert_points(self, collection_name: str, points: list) -> bool:
                # Qdrant-specific implementation...
                self.client.upsert(collection_name=collection_name, points=points)
                return True
        ```
        </details>
- [x] ~~**Update Standalone Scripts (`reindex_anime_database.py`):**~~
  - ~~Scripts that perform indexing will be updated to reflect the new architecture.~~
  - ~~They will become responsible for the full orchestration of vector generation: initializing the model factory, creating the models and processors, generating vectors for each entry, constructing the `PointStruct` objects, and calling the new pure `upsert_points` method on the `QdrantClient`.~~
  - ~~This moves the high-level business logic out of the reusable client and into the top-level application or script.~~

- [x] ~~**Architectural Refactoring (`vector_processing`):** Decouple embedding model selection and instantiation from the `TextProcessor` and `VisionProcessor`.~~
  - [x] ~~Create the new directory structure and files:~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/__init__.py`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/factory.py`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/text/base.py`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/text/fastembed_model.py`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/text/hf_model.py`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/vision/base.py`~~
    - [x] ~~`libs/vector_processing/src/vector_processing/embedding_models/vision/openclip_model.py`~~
  - [x] ~~In `base.py` files, define abstract base classes for text and vision models (e.g., `TextEmbeddingModel`) to establish a common interface.~~
  - [x] ~~Move the logic for specific model providers (e.g., FastEmbed, HuggingFace, OpenCLIP) into the concrete implementation classes (`fastembed_model.py`, etc.).~~
  - [x] ~~Implement the `EmbeddingModelFactory` in `factory.py` to read application settings and return the appropriate initialized model instance.~~
  - [x] ~~Refactor `TextProcessor` and `VisionProcessor` to align with the Single Responsibility Principle.~~
    - ~~**Decoupling:** The processors will be decoupled from model creation, lifecycle, and provider-specific encoding logic.~~
    - ~~**Dependency Injection:** The processors will receive a pre-initialized, generic model object (e.g., an object that conforms to the `TextEmbeddingModel` interface) in their constructor. They will no longer create the models themselves.~~
    - ~~**New Single Responsibility:** The processor's sole responsibility will be to orchestrate the high-level business logic of turning an `AnimeEntry` into a dictionary of named vectors. It will use a field mapper to get the data and then delegate the actual encoding task to the injected model object.~~
  - [x] ~~\<details>~~
        ~~\<summary>Example Refactoring for TextProcessor\</summary>~~

        ~~The `TextEmbeddingModel` abstract base class defines the contract for all text models:~~
        ~~```python~~
        ~~# In libs/vector_processing/src/vector_processing/embedding_models/text/base.py~~
        ~~from abc import ABC, abstractmethod~~
        ~~class TextEmbeddingModel(ABC):~~
            ~~@abstractmethod~~
            ~~def encode(self, texts: list[str]) -> list[list[float]]:~~
                ~~pass~~
            ~~# Other common properties like embedding_size can also be defined.~~
        ~~```~~

        ~~A concrete implementation for a specific provider:~~
        ~~```python~~
        ~~# In libs/vector_processing/src/vector_processing/embedding_models/text/fastembed_model.py~~
        ~~from .base import TextEmbeddingModel~~
        ~~from fastembed import TextEmbedding~~
        ~~class FastEmbedModel(TextEmbeddingModel):~~
            ~~def __init__(self, model_name: str, **kwargs):~~
                ~~self.model = TextEmbedding(model_name, **kwargs)~~

            ~~def encode(self, texts: list[str]) -> list[list[float]]:~~
                ~~# Logic specific to FastEmbed~~
                ~~return [e.tolist() for e in self.model.embed(texts)]~~
        ~~```~~

        ~~The refactored `TextProcessor` becomes much simpler and focuses only on its core task:~~
        ~~```python~~
        ~~# In libs/vector_processing/src/vector_processing/processors/text_processor.py~~
        ~~from ..embedding_models.text.base import TextEmbeddingModel~~
        ~~from .anime_field_mapper import AnimeFieldMapper~~

        ~~class TextProcessor:~~
            ~~# Receives dependencies, doesn't create them~~
            ~~def __init__(self, model: TextEmbeddingModel, field_mapper: AnimeFieldMapper):~~
                ~~self.model = model~~
                ~~self.field_mapper = field_mapper~~

            ~~def process_anime_vectors(self, anime: AnimeEntry) -> Dict[str, List[float]]:~~
                ~~# 1. Use self.field_mapper to get text data from anime object~~
                ~~# 2. Apply any specific preprocessing for that field~~
                ~~# 3. Delegate encoding to the injected model~~
                ~~#    embedding = self.model.encode([text_to_encode])[0]~~
                ~~# 4. Return the dictionary of vector names to embeddings~~
                ~~...~~
        ~~```~~
        ~~\</details>~~
  - [x] ~~**Refactor `VisionProcessor` with the same principles:**~~
    - ~~The `VisionProcessor` class currently has even more responsibilities than the `TextProcessor`, including model management, image encoding, image downloading/caching, and specialized similarity calculations. These will be separated:~~
    - ~~**Model Logic:** Will be extracted into a `VisionEmbeddingModel` abstract base class and a concrete `OpenClipModel` class within `libs/vector_processing/src/vector_processing/embedding_models/vision/`.~~
    - ~~**Image Downloading/Caching:** The logic for fetching and caching images from URLs (`_download_and_cache_image`, etc.) is a generic utility. It will be extracted into a separate, reusable `ImageDownloader` or `ImageCache` class, likely within `libs/common` or a new `libs/utils` library.~~
    - ~~**Specialized Similarity (`ccips`):** As already noted in a separate task, the `calculate_character_similarity` method will be moved to the `libs/enrichment` library, as it represents a specific enrichment step, not general vector processing.~~
    - ~~**`VisionProcessor`'s New Role:** The refactored class will be a lean orchestrator. It will receive a `VisionEmbeddingModel` and an `ImageDownloader` via dependency injection. Its sole responsibility will be to use these dependencies to manage the high-level business logic of creating `image_vector` and `character_image_vector` for an `AnimeEntry`.~~

- [x] ~~**Architectural Refactoring (`ccips` logic):** Move the `ccips` logic from `libs/vector_processing` into `libs/enrichment` to break the dependency, as discussed.~~
- [ ] **Test Restructuring:** Reorganize the top-level `tests/` directory to mirror the new `apps/` and `libs/` structure to keep tests logically co-located with the code they cover.
- [ ] Update all import statements in the moved Python files to reflect the new modular structure.
- [ ] Update `docker/Dockerfile.dev` to work with the new `apps/agent` structure.
- [ ] Update `docker/Dockerfile.prd` to work with the new `apps/agent` structure.
- [ ] Update `.github/workflows/claude.yml` to accommodate the monorepo structure for tests and linting.
- [ ] Run `pants check ::` to verify dependency resolution and code health.
- [ ] Run `pants lint ::` and `pants format ::` to ensure code quality.
- [ ] Run `pants test ::` to ensure all tests pass after the refactoring.

## Phase 5: Potential Challenges & Mitigations

- **Challenge: Circular Dependencies & Import Errors.**
  - **Mitigation:** The "bottom-up" migration strategy is designed to prevent this. The strict separation of `libs` (which cannot import from `apps`) and `apps` (which can import from `libs`) provides a clear architectural guardrail. Frequent use of `uv pip check` and `mypy` will identify these issues immediately.

- **Challenge: Test Discovery and Execution.**
  - **Mitigation:** By restructuring the `tests/` directory to mirror the monorepo layout (as noted in Phase 4), `pytest` should still be able to discover and run all tests from the root. We will verify this as part of the test restructuring task.

## Phase 6: Final Cleanup

- [ ] Delete `MONOREPO_MIGRATION_PLAN.md`.
- [ ] Commit the changes.
