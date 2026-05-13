FROM python:3.11-slim

WORKDIR /app

# System deps for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and source first for layer caching
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

# Copy scripts last so changes to them don't bust the install layer
COPY scripts ./scripts

ENTRYPOINT ["python"]