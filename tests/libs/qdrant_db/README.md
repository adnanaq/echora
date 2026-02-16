# Qdrant Client Tests

## Unit Tests
Unit tests now cover strict contract behavior for:
- `SearchRequest`-driven search
- Batch vector updates (`last-wins`, `fail`)
- Batch payload updates (`merge`, `overwrite`)
- Retry utility transient detection and backoff behavior

Run:
```bash
./pants test tests/libs/qdrant_db/unit:: --test-use-coverage=false
```

## Integration Tests
Integration tests in `tests/libs/qdrant_db/integration/` require:
- Running Qdrant database (default: `http://localhost:6333`)
- ML models downloaded (BGE-M3, OpenCLIP)
- Seed data file: `./assets/seed_data/anime_database.json`

Run:
```bash
./pants test tests/libs/qdrant_db/integration:: --test-use-coverage=false
```

## Notes
This codebase now uses a strict, exception-driven API contract in `qdrant_db.client`.
If you are migrating older test scenarios, use `SearchRequest`, `BatchVectorUpdateItem`, and `BatchPayloadUpdateItem`.
