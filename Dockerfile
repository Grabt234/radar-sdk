FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install git, curl, C++ compiler, and graphics development headers
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1-mesa-dev \
    libx11-dev \
    libcairo2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app



RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

    # Clone the specific branch
RUN git clone -b minor/genetic-optimiser-support --depth 1 https://github.com/Grabt234/radar-sdk.git .
# Sync dependencies (this will now compile glcontext and pycairo successfully)
RUN uv sync --frozen

CMD ["uv", "run", "tmp2.py"]