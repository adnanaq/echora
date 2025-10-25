<!--
description: Document major failure points in this project and they were solved. To be filled by AI.
-->

# Error Documentation & Fix History

## Overview

This document tracks errors, issues, and their resolutions discovered during development of the Anime Vector Service. It serves as a knowledge base to prevent recurring issues and improve development efficiency.

## Error Categories

### 1. Configuration Errors

#### Model Loading Failures

**Error**: `OSError: [Errno 28] No space left on device` during model download

- **Context**: HuggingFace model cache filling up available disk space
- **Root Cause**: Default cache directory on system partition with limited space
- **Resolution**: Configure `MODEL_CACHE_DIR` environment variable to point to larger storage
- **Prevention**: Always check available disk space before model downloads
- **Code Fix**: Added cache directory validation in settings configuration
- **Date**: 2025-08-04

#### Qdrant Connection Issues

**Error**: `ConnectionError: Cannot connect to Qdrant at http://localhost:6333`

- **Context**: Service startup failing when Qdrant not available
- **Root Cause**: Service starting before Qdrant database initialization
- **Resolution**: Added health check with retry logic in lifespan manager
- **Prevention**: Always use Docker Compose depends_on with health checks
- **Code Fix**: Implemented exponential backoff retry in QdrantClient initialization
- **Date**: 2025-08-03

### 2. Performance Issues

#### Memory Exhaustion

**Error**: `CUDA out of memory` during batch image processing

- **Context**: Processing large batches of high-resolution images
- **Root Cause**: JinaCLIP model loading multiple images simultaneously on GPU
- **Resolution**: Implemented batch size limiting and GPU memory management
- **Prevention**: Monitor GPU memory usage and implement batch size auto-tuning
- **Code Fix**: Added `max_batch_size` configuration and memory monitoring
- **Date**: 2025-08-02

#### Slow First Request

**Error**: 15-second response time on first API call after service start

- **Context**: Cold start performance causing timeout errors
- **Root Cause**: Models loaded lazily on first request instead of service startup
- **Resolution**: Added optional model warm-up during service initialization
- **Prevention**: Configure `MODEL_WARM_UP=true` for production deployments
- **Code Fix**: Implemented model pre-loading in lifespan startup
- **Date**: 2025-08-01

### 3. API and Validation Errors

#### Request Validation Failures

**Error**: `422 Unprocessable Entity` for valid image search requests

- **Context**: Base64 image data validation failing unexpectedly
- **Root Cause**: Pydantic model not handling data URL prefixes properly
- **Resolution**: Added custom validator to strip data URL prefixes
- **Prevention**: Include data URL handling in all image input validators
- **Code Fix**: Updated `ImageSearchRequest` model with custom validation
- **Date**: 2025-07-30

#### Vector Dimension Mismatch

**Error**: `ValueError: Vector dimension mismatch: expected 384, got 512`

- **Context**: Uploading image vectors to text vector collection
- **Root Cause**: Incorrect vector type targeting in multi-vector setup
- **Resolution**: Added vector type validation and proper named vector routing
- **Prevention**: Always specify vector name in multi-vector operations
- **Code Fix**: Enhanced QdrantClient with vector type validation
- **Date**: 2025-07-29

### 4. Database and Storage Issues

#### Collection Creation Failures

**Error**: `CollectionCreationError: Collection already exists with different configuration`

- **Context**: Development environment with inconsistent collection schemas
- **Root Cause**: Collection schema changes without dropping existing collection
- **Resolution**: Added schema validation and automatic migration logic
- **Prevention**: Always use version-controlled collection configurations
- **Code Fix**: Implemented collection schema validation and recreation
- **Date**: 2025-07-28

#### Index Building Timeouts

**Error**: `TimeoutError: HNSW index build exceeded maximum time limit`

- **Context**: Large dataset index building during development
- **Root Cause**: Default HNSW parameters not optimized for large datasets
- **Resolution**: Tuned HNSW parameters and increased timeout limits
- **Prevention**: Configure HNSW parameters based on expected dataset size
- **Code Fix**: Added configurable HNSW parameters in settings
- **Date**: 2025-07-27

### 5. Docker and Deployment Issues

#### Container Build Failures

**Error**: `Docker build failed: unable to resolve package dependencies`

- **Context**: Multi-stage Docker build failing on dependency installation
- **Root Cause**: Missing system dependencies for machine learning packages
- **Resolution**: Added system package installation in Dockerfile
- **Prevention**: Always test Docker builds in clean environments
- **Code Fix**: Updated Dockerfile with required system dependencies
- **Date**: 2025-07-26

#### Volume Mount Permissions

**Error**: `PermissionError: cannot write to mounted volume directory`

- **Context**: Qdrant data persistence failing in Docker Compose
- **Root Cause**: Host directory permissions incompatible with container user
- **Resolution**: Fixed directory permissions and container user configuration
- **Prevention**: Use named volumes or ensure proper permission setup
- **Code Fix**: Updated docker-compose.yml with proper volume configuration
- **Date**: 2025-07-25

## Resolution Patterns

### Pattern 1: Configuration Validation

**Problem**: Runtime errors due to invalid configuration
**Solution Pattern**:

1. Add Pydantic field validators for all configuration parameters
2. Implement startup validation checks
3. Provide clear error messages with suggested fixes
4. Document configuration requirements

### Pattern 2: Resource Monitoring

**Problem**: Resource exhaustion causing service failures
**Solution Pattern**:

1. Add resource usage monitoring (memory, GPU, disk)
2. Implement automatic resource limiting
3. Add alerts before resource exhaustion
4. Graceful degradation when resources low

### Pattern 3: Dependency Health Checks

**Problem**: Service failures due to unavailable dependencies
**Solution Pattern**:

1. Implement health checks for all external dependencies
2. Add retry logic with exponential backoff
3. Provide meaningful error messages for dependency failures
4. Allow service to start in degraded mode when appropriate

### Pattern 4: Input Validation Enhancement

**Problem**: Unexpected input causing processing errors
**Solution Pattern**:

1. Use strict Pydantic models for all API inputs
2. Add custom validators for complex data types
3. Implement input sanitization and normalization
4. Provide clear validation error messages

## Prevention Strategies

### Development Practices

1. **Always test in clean environments** - Use Docker for consistent builds
2. **Monitor resource usage** - Track memory, CPU, and GPU usage during development
3. **Validate configurations early** - Check settings during service startup
4. **Use health checks everywhere** - Monitor all external dependencies
5. **Test error conditions** - Explicitly test failure scenarios

### Code Quality Measures

1. **Comprehensive error handling** - Catch and handle all expected errors
2. **Logging with context** - Include relevant context in error logs
3. **Input validation** - Validate all inputs at API boundaries
4. **Resource cleanup** - Ensure proper cleanup of resources on failures
5. **Documentation** - Document all error conditions and resolutions

### Deployment Safeguards

1. **Staged deployments** - Test in staging environments first
2. **Health monitoring** - Continuous health monitoring in production
3. **Resource limits** - Set appropriate resource limits for containers
4. **Backup strategies** - Regular backups of critical data
5. **Rollback procedures** - Quick rollback mechanisms for failed deployments

## Known Issues (Current)

### High Priority

1. **Memory Usage Optimization Needed**
   - Issue: Service uses 3.5GB RAM with both models loaded
   - Impact: Limits scalability and increases hosting costs
   - Status: Under investigation
   - Workaround: Use single model deployments for memory-constrained environments
   - Target Resolution: Phase 3 optimization sprint

2. **Cold Start Performance**
   - Issue: 15-second delay on first request after service restart
   - Impact: Poor user experience and timeout errors
   - Status: Partial fix implemented (model warm-up option)
   - Workaround: Enable model warm-up for production
   - Target Resolution: Complete optimization in current sprint

### Medium Priority

1. **Large Image Processing Timeouts**
   - Issue: Images >5MB cause request timeouts
   - Impact: Limited to smaller image inputs
   - Status: Documented limitation
   - Workaround: Client-side image resizing before upload
   - Target Resolution: Streaming image processing in Phase 3

2. **Batch Operation Error Handling**
   - Issue: Partial batch failures provide insufficient error details
   - Impact: Difficult to diagnose which items in batch failed
   - Status: Documented for improvement
   - Workaround: Process smaller batches for better error isolation
   - Target Resolution: Enhanced error reporting in Phase 3

### Low Priority

1. **Debug Log Volume**
   - Issue: Debug mode generates excessive log volume
   - Impact: Log storage costs and reduced performance
   - Status: Known limitation
   - Workaround: Use INFO log level in production
   - Target Resolution: Log level optimization and filtering

## Monitoring and Alerting

### Critical Alerts

- Service health check failures
- Database connection losses
- Memory usage >90%
- Response time >5 seconds
- Error rate >1%

### Warning Alerts

- Memory usage >75%
- Response time >2 seconds
- Disk space <10GB
- Queue depth >100 items
- Error rate >0.5%

### Metrics to Track

- Request response times (p50, p95, p99)
- Memory and CPU usage
- Database connection health
- Model loading times
- Error rates by endpoint
- Concurrent request counts

This error documentation will be updated as new issues are discovered and resolved during ongoing development.
