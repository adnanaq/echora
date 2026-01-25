# QdrantClient Tests

## Integration Tests

**File**: `test_qdrant_client_integration.py`

All tests in this file are **integration tests** that require:
- Running Qdrant database (default: http://localhost:6333)
- ML models downloaded (BGE-M3, OpenCLIP)
- Test data file: `./assets/seed_data/anime_database.json`

### Running Integration Tests

```bash
# Run all integration tests
pytest -m integration

# Skip integration tests (run only unit tests)
pytest -m "not integration"

# Run specific integration test
pytest tests/vector/client/test_qdrant_client_integration.py::test_name
```

### Why Tests May Skip

Tests will automatically skip if:
- Qdrant database is not reachable
- Database connection fails during setup
- Required data files are missing

This is expected behavior - integration tests require infrastructure.

### Setting Up for Integration Tests

1. **Start Qdrant**:
   ```bash
   docker compose -f docker/docker-compose.yml up -d qdrant
   ```

2. **Verify Connection**:
   ```bash
   curl http://localhost:6333/health
   ```

3. **Run Tests**:
   ```bash
   pytest tests/vector/client/test_qdrant_client_integration.py --no-cov
   ```

## Test Coverage

- **36 integration tests** covering:
  - Vector update operations (single and batch)
  - Deduplication policies
  - Error handling and retries
  - Validation
  - End-to-end workflows with embedding generation

- **No unit tests yet** - all current tests require database
  - Future: Add unit tests with mocked AsyncQdrantClient
  - Future: Add unit tests for validation logic
