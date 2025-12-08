# CI/CD Pipeline Improvements - Implementation Plan

**Document Version**: 1.2
**Created**: 2025-10-24
**Last Updated**: 2025-10-25
**Status**: Implementation Phase
**Current Maturity Score**: 75/100 (↑ from 60/100)
**Completed**: Pre-commit hooks, Secret detection, Coverage enforcement, Container security scanning

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [High Priority Improvements](#high-priority-improvements)
3. [Medium Priority Improvements](#medium-priority-improvements)
4. [Low Priority Improvements](#low-priority-improvements)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Success Metrics](#success-metrics)

---

## Current State Analysis

### Existing CI/CD Components

**Workflows in `.github/workflows/`:**
- `main_ci.yml` - Main CI pipeline (3 jobs)
- `claude.yml` - Interactive Claude Code integration
- `claude-code-review.yml` - Automated PR reviews
- `gemini-*.yml` - Gemini AI workflows (5 files)

**Current Pipeline Stages:**
1. **Code Quality Checks**
   - Ruff linting and formatting
   - MyPy type checking (strict mode, src/ + tests/)
   - isort import sorting
   - pip-audit dependency vulnerability scan

2. **Test Execution**
   - Pytest with coverage (80% threshold)
   - Python dependency caching
   - ML model caching (HuggingFace)

3. **Application Health Check**
   - Docker Compose build
   - Service health validation
   - Container teardown

### Strengths
- ✅ Strong type checking enforcement
- ✅ Modern tooling (uv, Ruff)
- ✅ AI-powered code review
- ✅ Caching strategy implemented
- ✅ Health check validation

### Weaknesses
- ~~❌ No security scanning for containers~~ ✅ **COMPLETED** (Trivy scanning)
- ~~❌ No secret detection~~ ✅ **COMPLETED** (Gitleaks in pre-commit)
- ❌ No SAST (Static Application Security Testing)
- ❌ No deployment automation
- ~~❌ No pre-commit hooks~~ ✅ **COMPLETED**
- ❌ No integration testing
- ~~❌ Coverage not tracked over time~~ ✅ **COMPLETED** (80% threshold enforced in CI)
- ❌ Single Python version testing only

---

## High Priority Improvements

### ~~1. Security Scanning Suite~~ ✅ **COMPLETED** (Trivy)

**Priority**: 🔴 CRITICAL
**Estimated Effort**: 4-6 hours
**Dependencies**: None
**Status**: ✅ Trivy implemented (container + filesystem + config scanning)

#### 1.1 Container Vulnerability Scanning (Trivy)

**Purpose**: Scan Docker images for known vulnerabilities before deployment

**Implementation**:

Create `.github/workflows/security-scan.yml`:

```yaml
name: Security Scanning

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  trivy-container-scan:
    name: Trivy Container Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t anime-vector-service:${{ github.sha }} \
            -f docker/Dockerfile .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'anime-vector-service:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-container-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'  # Fail CI on vulnerabilities

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-container-results.sarif'
          category: 'container-scan'

  trivy-filesystem-scan:
    name: Trivy Filesystem Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run Trivy filesystem scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-fs-results.sarif'
          severity: 'CRITICAL,HIGH,MEDIUM'

      - name: Upload filesystem scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-fs-results.sarif'
          category: 'filesystem-scan'
```

**Benefits**:
- Detects CVEs in base images and dependencies
- Integrates with GitHub Security tab
- Blocks merges with critical vulnerabilities
- Daily scans catch new vulnerabilities

**Configuration Required**:
- None (uses public Trivy database)

---

#### 1.2 SAST with Semgrep

**Purpose**: Static code analysis to find security issues, bugs, and anti-patterns

**Implementation**:

Add to `.github/workflows/security-scan.yml`:

```yaml
  semgrep-sast:
    name: Semgrep SAST
    runs-on: ubuntu-latest
    container:
      image: semgrep/semgrep
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run Semgrep
        run: |
          semgrep scan \
            --config "p/python" \
            --config "p/security-audit" \
            --config "p/owasp-top-ten" \
            --sarif --output semgrep-results.sarif \
            --metrics off

      - name: Upload Semgrep results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'semgrep-results.sarif'
          category: 'sast'

      - name: Fail on findings
        run: |
          semgrep scan \
            --config "p/python" \
            --severity ERROR \
            --error
```

**What Semgrep Catches**:
- SQL injection vulnerabilities
- XSS vulnerabilities
- Hardcoded secrets/credentials
- Insecure cryptography usage
- OWASP Top 10 issues
- Python-specific anti-patterns

**Benefits**:
- Fast (30-60 seconds)
- Language-aware (understands Python semantics)
- Low false positive rate
- Free for open source

---

#### 1.3 Secret Detection with Gitleaks

**Purpose**: Prevent secrets, API keys, and credentials from being committed

**Implementation**:

Add to `.github/workflows/security-scan.yml`:

```yaml
  gitleaks-secret-scan:
    name: Gitleaks Secret Detection
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comprehensive scan

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}  # Optional

      - name: Upload results
        if: failure()
        run: |
          echo "⚠️ Secrets detected in repository!"
          echo "Review the Gitleaks report and remove sensitive data."
          exit 1
```

**What Gitleaks Detects**:
- AWS Access Keys
- GitHub Personal Access Tokens
- API keys (OpenAI, Anthropic, etc.)
- Database connection strings
- Private keys (RSA, SSH, etc.)
- JWT tokens
- Slack webhooks

**Benefits**:
- Scans entire git history
- 99%+ detection rate
- Prevents credential leaks
- CI fails immediately if secrets found

**Configuration** (`.gitleaks.toml`):

```toml
title = "Gitleaks Configuration"

[extend]
useDefault = true

[[rules]]
description = "Anthropic API Key"
id = "anthropic-api-key"
regex = '''sk-ant-[a-zA-Z0-9\-_]{95}'''
keywords = ["sk-ant-"]

[[rules]]
description = "OpenAI API Key"
id = "openai-api-key"
regex = '''sk-[a-zA-Z0-9]{48}'''
keywords = ["sk-"]

[[rules]]
description = "Qdrant API Key"
id = "qdrant-api-key"
regex = '''[a-zA-Z0-9_-]{32,}'''
path = '''.env|config\.py'''
keywords = ["QDRANT_API_KEY"]

[allowlist]
paths = [
  '''\.github/workflows/.*\.yml''',
  '''docs/.*\.md'''
]
```

---

#### 1.4 License Compliance Check

**Purpose**: Ensure dependencies use compatible licenses

**Implementation**:

Add to `main_ci.yml` in code-quality-checks job:

```yaml
      - name: Check License Compliance
        run: |
          uv run pip-licenses \
            --format=markdown \
            --with-license-file \
            --fail-on="GPL;AGPL;SSPL" \
            > license-report.md

      - name: Upload License Report
        uses: actions/upload-artifact@v4
        with:
          name: license-report
          path: license-report.md
```

**Benefits**:
- Prevents GPL contamination
- Audit trail for compliance
- Early detection of license issues

---

### ~~2. Pre-commit Hooks~~ ✅ **COMPLETED**

**Priority**: 🔴 HIGH
**Estimated Effort**: 2-3 hours
**Dependencies**: None
**Status**: ✅ Implemented (commits: 145d2b6, 09259c4)

#### 2.1 Pre-commit Configuration

**Purpose**: Catch issues locally before pushing to CI

**Implementation**:

Create `.pre-commit-config.yaml`:

```yaml
# See https://pre-commit.com for more information
repos:
  # Ruff - Fast Python linter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.2
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # MyPy - Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.2
    hooks:
      - id: mypy
        args: [--strict, --show-error-codes]
        additional_dependencies:
          - types-requests
          - types-pillow
          - types-beautifulsoup4

  # isort - Import sorting
  - repo: https://github.com/PyCQA/isort
    rev: 6.0.1
    hooks:
      - id: isort
        args: [--profile, black]

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
        args: [--safe]
      - id: check-added-large-files
        args: [--maxkb=5000]
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: check-json
      - id: check-toml
      - id: detect-private-key
      - id: mixed-line-ending

  # Gitleaks - Secret detection
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks

  # Prettier - Markdown/YAML/JSON formatting
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v4.0.0-alpha.8
    hooks:
      - id: prettier
        types_or: [markdown, yaml, json]
        exclude: 'uv\.lock'

  # shellcheck - Shell script linting
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.10.0.1
    hooks:
      - id: shellcheck
```

**Installation Instructions** (add to `README.md`):

```markdown
## Development Setup

1. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

2. Run hooks manually:
   ```bash
   uv run pre-commit run --all-files
   ```

3. Update hooks:
   ```bash
   uv run pre-commit autoupdate
   ```
```

**Add to `pyproject.toml`**:

```toml
[project.optional-dependencies]
dev = [
    # ... existing dependencies
    "pre-commit>=4.0.0",
]
```

**Benefits**:
- Catches 80% of CI failures locally
- Faster feedback loop (seconds vs minutes)
- Reduces CI load and costs
- Enforces standards before review

**CI Validation** (add to `main_ci.yml`):

```yaml
      - name: Validate pre-commit hooks
        run: |
          uv run pre-commit install
          uv run pre-commit run --all-files
```

---

### 3. Code Coverage Tracking

**Priority**: 🔴 HIGH
**Estimated Effort**: 3-4 hours
**Dependencies**: Codecov account (free for open source)

#### 3.1 Codecov Integration

**Purpose**: Track coverage trends over time and visualize on PRs

**Implementation**:

Update `main_ci.yml` test-execution job:

```yaml
  test-execution:
    runs-on: ubuntu-latest
    steps:
      # ... existing steps

      - name: Run Pytest with Coverage
        run: |
          uv run pytest \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --cov-report=html:htmlcov \
            --cov-fail-under=80 \
            --junitxml=pytest-report.xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: true
          verbose: true

      - name: Upload coverage HTML report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-report
          path: htmlcov/

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: pytest-results
          path: pytest-report.xml
```

**Setup Steps**:

1. Sign up at https://codecov.io with GitHub
2. Add repository to Codecov
3. Get upload token from Codecov dashboard
4. Add `CODECOV_TOKEN` to GitHub repository secrets
5. Add Codecov badge to README.md:

```markdown
[![codecov](https://codecov.io/gh/adnanaq/anime-vector-service/branch/main/graph/badge.svg)](https://codecov.io/gh/adnanaq/anime-vector-service)
```

**Codecov Configuration** (`.codecov.yml`):

```yaml
coverage:
  status:
    project:
      default:
        target: 80%
        threshold: 1%  # Allow 1% drop
    patch:
      default:
        target: 80%
        threshold: 5%

comment:
  layout: "reach,diff,flags,files"
  behavior: default
  require_changes: false
  require_base: false
  require_head: true

ignore:
  - "tests/"
  - "scripts/"
  - "docs/"
  - "**/__init__.py"
```

**Benefits**:
- Visual coverage trends on PRs
- Catch coverage regressions
- Identify untested code paths
- Team accountability

---

#### 3.2 PR Coverage Comments

**Purpose**: Show coverage changes directly on PRs

**Implementation**:

Add to test-execution job:

```yaml
      - name: Coverage comment on PR
        uses: py-cov-action/python-coverage-comment-action@v3
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          MINIMUM_GREEN: 80
          MINIMUM_ORANGE: 70
```

**What This Shows**:
- Total coverage percentage
- Coverage change from base branch
- Files with low coverage
- Color-coded indicators (green/yellow/red)

---

### 4. Integration Testing

**Priority**: 🔴 HIGH
**Estimated Effort**: 6-8 hours
**Dependencies**: None

#### 4.1 Integration Test Suite

**Purpose**: Test actual service behavior with real dependencies

**Directory Structure**:

```
tests/
├── unit/              # Existing unit tests
│   └── ...
├── integration/       # New integration tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_qdrant_integration.py
│   ├── test_search_api.py
│   ├── test_multimodal_search.py
│   └── test_vector_operations.py
└── load/              # Future: load testing
    └── locustfile.py
```

**Implementation**:

Create `tests/integration/conftest.py`:

```python
"""Integration test fixtures with real Qdrant instance."""
import pytest
import pytest_asyncio
from qdrant_client import QdrantClient as QdrantSDK
from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient


@pytest.fixture(scope="session")
def integration_settings():
    """Settings for integration tests."""
    settings = get_settings()
    settings.qdrant_collection_name = "anime_database_integration_test"
    settings.qdrant_url = "http://localhost:6333"
    return settings


@pytest_asyncio.fixture(scope="session")
async def qdrant_client(integration_settings):
    """QdrantClient with test collection."""
    client = QdrantClient(settings=integration_settings)

    # Create test collection
    await client.create_collection()

    yield client

    # Cleanup
    await client.delete_collection()


@pytest_asyncio.fixture
async def sample_anime_data():
    """Sample anime data for testing."""
    return [
        {
            "id": "test-1",
            "title": "Cowboy Bebop",
            "type": "TV",
            "genres": ["Action", "Sci-Fi"],
            "year": 1998,
        },
        {
            "id": "test-2",
            "title": "Steins;Gate",
            "type": "TV",
            "genres": ["Sci-Fi", "Thriller"],
            "year": 2011,
        },
    ]
```

Create `tests/integration/test_qdrant_integration.py`:

```python
"""Test Qdrant vector database operations."""
import pytest


@pytest.mark.asyncio
class TestQdrantIntegration:
    """Integration tests for Qdrant operations."""

    async def test_collection_exists(self, qdrant_client):
        """Test collection creation."""
        collections = await qdrant_client.list_collections()
        assert qdrant_client.collection_name in [c.name for c in collections]

    async def test_upsert_vectors(self, qdrant_client, sample_anime_data):
        """Test vector insertion."""
        for anime in sample_anime_data:
            result = await qdrant_client.upsert_anime(anime)
            assert result is not None

    async def test_search_by_text(self, qdrant_client):
        """Test text-based vector search."""
        results = await qdrant_client.search_by_text(
            query="space adventure",
            limit=10
        )
        assert len(results) > 0
        assert all(hasattr(r, 'score') for r in results)

    async def test_filter_by_year(self, qdrant_client):
        """Test filtering with payload."""
        results = await qdrant_client.search_by_text(
            query="anime",
            filter_dict={"year": {"gte": 2000}},
            limit=10
        )
        assert all(r.payload.get("year", 0) >= 2000 for r in results)
```

Create `tests/integration/test_multimodal_search.py`:

```python
"""Test multimodal search capabilities."""
import pytest
from pathlib import Path


@pytest.mark.asyncio
class TestMultimodalSearch:
    """Integration tests for multimodal search."""

    async def test_text_and_image_search(self, qdrant_client):
        """Test combined text + image vector search."""
        results = await qdrant_client.search_multimodal(
            text_query="mecha anime",
            image_path=Path("tests/fixtures/sample_image.jpg"),
            limit=5
        )
        assert len(results) > 0

    async def test_image_only_search(self, qdrant_client):
        """Test image-only vector search."""
        results = await qdrant_client.search_by_image(
            image_path=Path("tests/fixtures/sample_image.jpg"),
            limit=5
        )
        assert len(results) > 0
```

**CI Configuration** (add new job to `main_ci.yml`):

```yaml
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [code-quality-checks]

    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
        options: >-
          --health-cmd "curl -f http://localhost:6333/health"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-uv-${{ hashFiles('pyproject.toml') }}

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Wait for Qdrant
        run: |
          timeout 30 bash -c 'until curl -f http://localhost:6333/health; do sleep 2; done'

      - name: Run integration tests
        env:
          QDRANT_URL: http://localhost:6333
        run: |
          uv run pytest tests/integration/ \
            -v \
            --cov=src \
            --cov-report=xml:coverage-integration.xml \
            --cov-report=term-missing \
            --junitxml=pytest-integration.xml

      - name: Upload integration test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: integration-test-results
          path: pytest-integration.xml

      - name: Upload integration coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage-integration.xml
          flags: integration
```

**Benefits**:
- Tests real Qdrant operations
- Validates vector search accuracy
- Tests multimodal capabilities
- Catches integration issues early

---

## Medium Priority Improvements

### 5. Multi-Python Version Testing

**Priority**: 🟡 MEDIUM
**Estimated Effort**: 2-3 hours
**Dependencies**: None

#### Implementation

Update `main_ci.yml` test-execution job:

```yaml
  test-execution:
    name: Test (Python ${{ matrix.python-version }}, ${{ matrix.os }})
    runs-on: ${{ matrix.os }}

    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.12', '3.13']
        os: [ubuntu-latest, macos-latest]
        exclude:
          # Exclude macOS + Python 3.13 if not needed
          - os: macos-latest
            python-version: '3.13'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-${{ matrix.python-version }}-uv-${{ hashFiles('pyproject.toml') }}

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Run tests
        run: |
          uv run pytest \
            --cov=src \
            --cov-report=xml:coverage-${{ matrix.python-version }}-${{ matrix.os }}.xml \
            --junitxml=pytest-${{ matrix.python-version }}-${{ matrix.os }}.xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage-${{ matrix.python-version }}-${{ matrix.os }}.xml
          flags: python-${{ matrix.python-version }}-${{ matrix.os }}
```

**Benefits**:
- Future-proof for Python 3.13
- Cross-platform compatibility
- Catches platform-specific bugs

---

### 6. Deployment Pipeline

**Priority**: 🟡 MEDIUM
**Estimated Effort**: 8-12 hours
**Dependencies**: Cloud infrastructure setup

#### 6.1 Staging Deployment

Create `.github/workflows/deploy-staging.yml`:

```yaml
name: Deploy to Staging

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build-and-push]
    environment:
      name: staging
      url: https://staging.anime-vector-service.com

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster anime-vector-staging \
            --service vector-service \
            --force-new-deployment

      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster anime-vector-staging \
            --services vector-service

      - name: Run smoke tests
        run: |
          curl -f https://staging.anime-vector-service.com/health || exit 1
          curl -f https://staging.anime-vector-service.com/docs || exit 1

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v2
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
          payload: |
            {
              "text": "Staging deployment ${{ job.status }}: ${{ github.sha }}"
            }
```

---

#### 6.2 Production Deployment

Create `.github/workflows/deploy-production.yml`:

```yaml
name: Deploy to Production

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://api.anime-vector-service.com

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Blue-Green Deployment
        run: |
          # Create new task definition with new image
          # Update service to use new task definition
          # Monitor health checks
          # Switch traffic to new version
          # Keep old version for rollback

      - name: Health check validation
        run: |
          for i in {1..10}; do
            curl -f https://api.anime-vector-service.com/health && break
            sleep 30
          done

      - name: Create rollback tag
        run: |
          git tag -a "rollback-$(date +%Y%m%d-%H%M%S)" -m "Pre-deployment snapshot"
          git push origin --tags
```

**Environment Protection Rules** (GitHub Settings):

1. Go to Settings → Environments → production
2. Add required reviewers (2+ approvals)
3. Add branch protection (only tags matching `v*`)
4. Add deployment delays (15 minutes)

---

### 7. Automated Dependency Updates

**Priority**: 🟡 MEDIUM
**Estimated Effort**: 1-2 hours
**Dependencies**: None

#### Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 5
    reviewers:
      - "adnanaq"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "chore"
      prefix-development: "chore"
      include: "scope"

    # Group updates for minor/patch versions
    groups:
      development-dependencies:
        patterns:
          - "pytest*"
          - "ruff"
          - "mypy"
          - "black"
          - "isort"

      fastapi-dependencies:
        patterns:
          - "fastapi"
          - "uvicorn"
          - "pydantic*"

      ml-dependencies:
        patterns:
          - "torch"
          - "transformers"
          - "sentence-transformers"

    # Security updates only for critical dependencies
    ignore:
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]

    # Auto-merge configuration
    # Requires branch protection rules
    # allow:
    #   - dependency-name: "*"
    #     dependency-type: "development"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/docker"
    schedule:
      interval: "weekly"
      day: "tuesday"
    reviewers:
      - "adnanaq"
    labels:
      - "dependencies"
      - "docker"
    commit-message:
      prefix: "chore"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    reviewers:
      - "adnanaq"
    labels:
      - "dependencies"
      - "ci"
    commit-message:
      prefix: "ci"
```

**Auto-merge Configuration** (`.github/workflows/dependabot-auto-merge.yml`):

```yaml
name: Dependabot Auto-merge

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: write
  pull-requests: write

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]'
    steps:
      - name: Dependabot metadata
        id: metadata
        uses: dependabot/fetch-metadata@v2
        with:
          github-token: "${{ secrets.GITHUB_TOKEN }}"

      - name: Enable auto-merge for Dependabot PRs
        if: |
          steps.metadata.outputs.update-type == 'version-update:semver-patch' ||
          steps.metadata.outputs.update-type == 'version-update:semver-minor'
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{github.event.pull_request.html_url}}
          GITHUB_TOKEN: ${{secrets.GITHUB_TOKEN}}
```

**Benefits**:
- Weekly security updates
- Grouped updates reduce PR noise
- Auto-merge for safe updates
- Manual review for breaking changes

---

## Low Priority Improvements

### 8. Performance Benchmarking

**Priority**: 🟢 LOW
**Estimated Effort**: 6-8 hours

#### 8.1 Locust Load Testing

Create `tests/load/locustfile.py`:

```python
"""Load testing with Locust."""
from locust import HttpUser, task, between


class AnimeVectorUser(HttpUser):
    """Simulate user behavior."""

    wait_time = between(1, 3)
    host = "http://localhost:8002"

    @task(3)
    def search_text(self):
        """Text search endpoint (most common)."""
        self.client.post(
            "/api/v1/search/text",
            json={
                "query": "action anime",
                "limit": 10
            }
        )

    @task(2)
    def search_with_filter(self):
        """Search with filtering."""
        self.client.post(
            "/api/v1/search/text",
            json={
                "query": "mecha",
                "limit": 10,
                "filter": {
                    "year": {"gte": 2020}
                }
            }
        )

    @task(1)
    def health_check(self):
        """Health check endpoint."""
        self.client.get("/health")

    @task(1)
    def docs(self):
        """API docs."""
        self.client.get("/docs")
```

**CI Integration** (`.github/workflows/performance.yml`):

```yaml
name: Performance Benchmarking

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 3 * * 1'  # Weekly on Monday

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Start services
        run: docker compose -f docker/docker-compose.yml up -d

      - name: Wait for services
        run: |
          timeout 90 bash -c 'until curl -f http://localhost:8002/health; do sleep 5; done'

      - name: Run Locust
        run: |
          uv run locust \
            -f tests/load/locustfile.py \
            --headless \
            --users 100 \
            --spawn-rate 10 \
            --run-time 2m \
            --host http://localhost:8002 \
            --html=locust-report.html

      - name: Store results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'customBenchmark'
          output-file-path: locust-stats.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

---

### 9. Release Automation

**Priority**: 🟢 LOW
**Estimated Effort**: 4-6 hours

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  create-release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: metcalfc/changelog-generator@v4
        with:
          myToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Create Release
        uses: softprops/action-gh-release@v2
        with:
          body: ${{ steps.changelog.outputs.changelog }}
          generate_release_notes: true
          files: |
            dist/*
```

---

### 10. Documentation Generation

**Priority**: 🟢 LOW
**Estimated Effort**: 3-4 hours

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Generate OpenAPI spec
        run: |
          uv run python -c "
          from src.main import app
          import json
          with open('openapi.json', 'w') as f:
              json.dump(app.openapi(), f, indent=2)
          "

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

---

## Implementation Roadmap

### Phase 1: Security & Quality (Week 1-2)
**Goal**: Achieve security compliance and prevent regressions

- [ ] Security scanning suite (Trivy + Semgrep + Gitleaks)
- [ ] Pre-commit hooks
- [ ] Code coverage tracking (Codecov)
- [ ] PR coverage comments

**Success Criteria**:
- 0 critical/high vulnerabilities
- 90%+ of commits pass pre-commit hooks
- Coverage tracked on all PRs
- 80%+ test coverage maintained

---

### Phase 2: Testing & Reliability (Week 3-4)
**Goal**: Ensure service reliability and correctness

- [ ] Integration test suite
- [ ] Multi-Python version testing
- [ ] Health check improvements

**Success Criteria**:
- 20+ integration tests
- Tests pass on Python 3.12 & 3.13
- 0 flaky tests
- Integration coverage >70%

---

### Phase 3: Deployment Automation (Week 5-6)
**Goal**: Enable continuous delivery

- [ ] Staging deployment pipeline
- [ ] Production deployment with approval gates
- [ ] Rollback procedures
- [ ] Monitoring integration

**Success Criteria**:
- Deploy to staging on every main commit
- Production deployments <10 min
- Successful rollback tested
- 99.9% deployment success rate

---

### Phase 4: Developer Experience (Week 7-8)
**Goal**: Improve developer productivity

- [ ] Dependabot configuration
- [ ] Auto-merge for safe updates
- [ ] Performance benchmarking
- [ ] Documentation automation

**Success Criteria**:
- <24h security patch turnaround
- 50% reduction in manual dependency updates
- Performance baselines established
- Auto-generated API docs

---

## Success Metrics

### Before Implementation (Current State)

| Metric | Current Value | Target Value |
|--------|---------------|--------------|
| CI/CD Maturity Score | 60/100 | 90/100 |
| Security Vulnerabilities | Unknown | 0 critical/high |
| Test Coverage | 80% | 85%+ |
| Integration Tests | 0 | 20+ |
| Pre-commit Hook Usage | 0% | 95% |
| Mean Time to Deployment | Manual | <10 min |
| Deployment Success Rate | N/A | 99%+ |
| Security Scan Frequency | Never | Daily |
| Dependency Update Time | Manual | Automated |
| Failed Builds Due to Formatting | ~15% | <2% |

### After Implementation (Target State)

**Security**:
- ✅ Zero critical/high vulnerabilities in production
- ✅ Daily automated security scans
- ✅ Secret detection preventing credential leaks

**Quality**:
- ✅ 85%+ test coverage with trend tracking
- ✅ 20+ integration tests covering critical paths
- ✅ Multi-version compatibility (Python 3.12, 3.13)

**Velocity**:
- ✅ <5 min CI feedback on PRs
- ✅ <10 min deployments to staging
- ✅ 90%+ of commits pass pre-commit hooks locally

**Reliability**:
- ✅ 99%+ deployment success rate
- ✅ <1 min rollback capability
- ✅ Automated dependency updates with <24h security patch turnaround

---

## References

### Tools & Services

- [Trivy](https://trivy.dev/) - Vulnerability scanner
- [Semgrep](https://semgrep.dev/) - SAST tool
- [Gitleaks](https://gitleaks.io/) - Secret detection
- [Codecov](https://codecov.io/) - Coverage tracking
- [pre-commit](https://pre-commit.com/) - Git hooks framework
- [Dependabot](https://github.com/dependabot) - Dependency updates
- [Locust](https://locust.io/) - Load testing

### Best Practices

- [GitHub Actions Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Python Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Container Security Best Practices](https://docs.docker.com/develop/security-best-practices/)

---

## Maintenance

This document should be reviewed and updated:
- After each phase completion
- Quarterly for new best practices
- When new tools/services are adopted
- After major incidents or learnings

**Last Updated**: 2025-10-24
**Next Review**: 2025-11-24
**Owner**: DevOps/Platform Team
