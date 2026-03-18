# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# System dependencies for OpenGL rendering (pyrender/open3d/vtk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    libegl1 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app

# ── Stage 1: resolve & install deps (cached unless lock/toml change) ──
COPY pyproject.toml pixi.lock ./
RUN pixi install --locked

# ── Stage 2: copy source & install CUDA extensions ──
COPY . .
RUN pixi install --locked && pixi run post-install

# Use EGL for headless GPU rendering (pyrender/open3d)
ENV PYOPENGL_PLATFORM=egl

ENTRYPOINT ["pixi", "run"]
CMD ["zerokey", "--help"]
