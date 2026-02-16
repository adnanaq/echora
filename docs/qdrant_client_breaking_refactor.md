# Qdrant Client Breaking Refactor

## Summary
The Qdrant client now uses strict request/response contracts and domain exceptions.

## New API Surface
- `search(request: SearchRequest) -> list[SearchHit]`
- `update_vectors(updates: list[BatchVectorUpdateItem], dedup_policy="last-wins"|"fail") -> BatchOperationResult`
- `update_payload(updates: list[BatchPayloadUpdateItem], mode="merge"|"overwrite", dedup_policy="last-wins"|"fail") -> BatchOperationResult`
- `add_documents(...) -> BatchOperationResult`

## Configuration Changes
`QdrantConfig` now includes explicit primary vector names:
- `primary_text_vector_name`
- `primary_image_vector_name`

Both must exist in `vector_names`.

## Behavior Changes
- No heuristic vector-role inference from vector names.
- No broad dedup policy matrix. Only `last-wins` and `fail` are supported.
- Retry transient detection now considers retryable HTTP status codes.
- Payload indexing is no longer implicitly run during collection init. Use `setup_payload_indexes()` explicitly.

## Migration Guide
1. Replace `search(text_embedding=..., image_embedding=..., ...)` with `search(SearchRequest(...))`.
2. Replace `update_single_point_vector`/`update_batch_point_vectors` with `update_vectors`.
3. Replace payload writes with `update_payload`.
4. Update config to define explicit primary vector names when custom vector names are used.
