# Dockerfile for Anime Vector Service
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv (modern fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (it's installed under /root/.local/bin)
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --system --no-dev --frozen

# Copy source code
COPY src/ ./src/
COPY README.md .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

# Command to run the application
CMD ["python", "-m", "src.main"]
