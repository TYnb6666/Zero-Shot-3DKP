# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# System dependencies for OpenGL rendering (pyrender/open3d/vtk)
RUN apt-get update && apt-get install -y build-essential \
    curl ca-certificates git \
    libegl1 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app

# Copy full source (pixi needs the local package for resolution)
COPY . .

# Tell pixi/conda that CUDA 13.0 is available (driver not visible at build time)
ENV CONDA_OVERRIDE_CUDA="13.0"

# Install all dependencies (conda + PyPI + CUDA extensions)
RUN pixi install --all

# Use EGL for headless GPU rendering (pyrender/open3d)
ENV PYOPENGL_PLATFORM=egl

ENTRYPOINT ["pixi", "run"]
CMD ["zerokey", "--help"]
