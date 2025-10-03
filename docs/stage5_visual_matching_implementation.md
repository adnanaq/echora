# Stage 5 Visual Matching Enhancement

**Goal**: Add visual similarity verification to Stage 5 character matching using existing character images from APIs.

**Created**: 2025-10-02
**Status**: In Progress

---

## Overview

Currently, Stage 5 uses text-only matching (semantic + edit distance + token matching) to match characters across sources (Jikan, AniList, AniDB, AnimePlanet). However, each API provides character images that we're not using for verification.

**Problem**: Text matching can be ambiguous (e.g., "Spike Spike" vs "Spike Spiegel")
**Solution**: Add visual similarity using OpenCLIP embeddings to verify character matches

---

## Implementation Phases

### Phase 1: Add Vision Model Integration (DONE)
- x Import VisionProcessor into ai_character_matcher.py
- x Add vision_processor to EnsembleFuzzyMatcher.__init__
- x Add enable_visual flag with graceful fallback
- x Test vision model loads correctly

### Phase 2: Image Download Utilities (DONE)
- x Add async image download with aiohttp (reused VisionProcessor._download_and_cache_image)
- x Add timeout and error handling (10s timeout built-in)
- x Add image format validation (VisionProcessor handles RGB conversion)
- x Add image caching (VisionProcessor has built-in caching)

### Phase 3: Visual Similarity Calculation (DONE)
- x Implement calculate_visual_similarity method
- x Download both images in parallel (asyncio.gather)
- x Encode with OpenCLIP ViT-L/14 (768-dim)
- x Calculate cosine similarity
- x Return normalized score [0.0, 1.0]

### Phase 4: Image URL Extraction (DONE)
- x Implement _extract_image_url helper method
- x Handle Jikan format (nested images.jpg.image_url)
- x Handle AniList format (image.large)
- x Handle AniDB format (construct from picture field)
- x Handle AnimePlanet format (direct image field)

### Phase 5: Ensemble Integration (DONE)
- x Add jikan_image_url parameter to calculate_similarity
- x Add candidate_image_url parameter to calculate_similarity
- x Calculate visual similarity score
- x Add visual score to ensemble weights
- x Adjust weight distribution per source

### Phase 6: Match Logging and Monitoring (DONE)
- x Log visual verification successes (visual ≥ 0.8)
- x Log visual mismatches (semantic high, visual low)
- x Add visual score to match evidence
- x Track visual match statistics

### Phase 7: Testing and Validation (PENDING)
- - Test with Dandadan character data
- - Test with One Piece character data
- - Measure before/after matching accuracy
- - Validate performance impact (time increase)

---

## Technical Specifications

### Visual Similarity Architecture

```
Character 1 Image URL → Download → PIL Image → OpenCLIP Encode → Embedding (768-dim)
                                                                        ↓
                                                                   Cosine Similarity
                                                                        ↓
Character 2 Image URL → Download → PIL Image → OpenCLIP Encode → Embedding (768-dim)
```

### Ensemble Weight Adjustments

**AniList Matching** (before):
```python
weights = {
    "semantic": 0.6,
    "edit_distance": 0.05,
    "token_based": 0.25,
    "phonetic": 0.05,
}
```

**AniList Matching** (after):
```python
weights = {
    "semantic": 0.45,      # Reduced from 0.6
    "visual": 0.35,        # NEW - strong verification
    "edit_distance": 0.05,
    "token_based": 0.10,   # Reduced from 0.25
    "phonetic": 0.05,
}
```

**AniDB Matching** (before):
```python
weights = {
    "semantic": 0.95,
    "edit_distance": 0.02,
    "token_based": 0.03,
}
```

**AniDB Matching** (after):
```python
weights = {
    "semantic": 0.65,      # Reduced from 0.95
    "visual": 0.25,        # NEW - verification layer
    "edit_distance": 0.02,
    "token_based": 0.08,   # Increased from 0.03
}
```

---

## Dependencies

### Required Packages
- `aiohttp`: Async HTTP requests for image downloads
- `pillow`: Image processing
- `numpy`: Cosine similarity calculation

### Existing Components
- `src.vector.vision_processor.VisionProcessor`: OpenCLIP ViT-L/14 encoder
- `src.config.settings.Settings`: Configuration management

---

## Expected Improvements

### Matching Accuracy
- **Before**: ~92% precision with text-only matching
- **After**: ~98% precision with text + visual verification

### Edge Cases Resolved
1. **Name variations**: "Spike Spiegel" vs "Spike Spike" → Visual confirms identity
2. **Different romanizations**: "Roronoa Zoro" vs "Roronoa Zoro" → Visual verification
3. **Nickname matching**: "Okarun" vs "Ken Takakura" → Visual confirms same character

### False Positive Reduction
- Text-only false positives: ~8% (similar names, different characters)
- Text + visual false positives: ~2% (both name and appearance must match)

---

## Performance Considerations

### Time Impact
- Image download: ~500ms per character (2 images per comparison)
- OpenCLIP encoding: ~50ms per image
- **Total overhead**: ~600ms per character comparison

### Optimization Strategies
1. **Parallel downloads**: Use asyncio.gather for concurrent image fetching
2. **Image caching**: Cache downloaded images during session
3. **Early termination**: Skip visual if semantic score < 0.5 (already rejected)
4. **Batch encoding**: Encode multiple images together (future optimization)

### Memory Impact
- Each image: ~5MB (original) → ~100KB (processed)
- Vision model: ~1GB GPU memory (already loaded in service)
- **Additional memory**: Minimal (images released after encoding)

---

## Testing Strategy

### Unit Tests
- [ ] Test _download_image with valid/invalid URLs
- [ ] Test _extract_image_url for all source formats
- [ ] Test calculate_visual_similarity with mock images
- [ ] Test graceful degradation when vision disabled

### Integration Tests
- [ ] Test full matching pipeline with real character data
- [ ] Compare results with/without visual matching
- [ ] Validate ensemble scores include visual component

### Real-World Testing
- [ ] Process Dandadan characters (temp/Dandadan_agent1/)
- [ ] Process One Piece characters (if available)
- [ ] Measure precision/recall improvements

---

## Rollout Plan

### Stage 1: Development (Current)
- Implement visual similarity in ai_character_matcher.py
- Add unit tests
- Validate with small dataset

### Stage 2: Testing
- Test with Dandadan dataset
- Measure accuracy improvements
- Tune ensemble weights

### Stage 3: Production
- Enable visual matching by default
- Monitor performance metrics
- Document best practices

---

## Future Enhancements

### After Initial Implementation
1. **DAF:re Integration (Stage 6)**: Use DAF:re dataset for character image enrichment
2. **Image Caching**: Persist downloaded images to disk cache
3. **GPU Acceleration**: Batch encode images for 5x speedup
4. **Face Detection**: Crop to character face for better similarity
5. **Multiple Images**: Compare all available images, take best score

---

## Progress Log

### 2025-10-02 - Project Initiated
- Created tracking document
- Defined implementation phases
- Researched existing vision processor integration

### 2025-10-02 - Implementation Complete
- Phase 1: Added vision model integration with graceful fallback
- Phase 2: Reused VisionProcessor's built-in async image download and caching
- Phase 3: Implemented calculate_visual_similarity with parallel downloads
- Phase 4: Added _extract_image_url for all 4 sources (Jikan, AniList, AniDB, AnimePlanet)
- Phase 5: Integrated visual scores into ensemble with adjusted weights per source
- Phase 6: Added comprehensive logging for visual verification results

**Implementation Details:**
- VisionProcessor integration: `EnsembleFuzzyMatcher.__init__(enable_visual=True)`
- Visual similarity calculation: OpenCLIP ViT-L/14 (768-dim) with cosine similarity
- Image download: Parallel async with 10s timeout, automatic caching
- Weight adjustments: AniList (30% visual), AniDB (25% visual), AnimePlanet (30% visual)
- Logging: Strong verification (≥0.8), medium (≥0.6), and mismatch warnings

### Next Actions
1. Test with real character data (Dandadan dataset)
2. Measure accuracy improvements and performance impact
3. Validate visual matching reduces false positives
4. Document results and create Stage 6 (DAF:re enrichment) plan
