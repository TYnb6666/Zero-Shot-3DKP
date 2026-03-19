# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

**ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models**

A research codebase implementing zero-shot 3D keypoint detection using Multi-Modal Large Language Models (MLLMs). The system extracts and names salient keypoints on 3D shapes without ground truth labels by leveraging pixel-level annotations from MLLMs (Molmo, GPT-4o) combined with feature backprojection from vision models (DINOv2, CLIP, SAM).

**Paper:** [arXiv:2412.06292](https://arxiv.org/abs/2412.06292)
**Website:** [sites.google.com/view/zerokey](https://sites.google.com/view/zerokey)

## Technology Stack

- **Python:** 3.13+
- **ML Framework:** PyTorch 2.10.0+cu130, PyTorch3D (gt4o4 fork)
- **CUDA:** 13.0
- **Vision Models:** DINOv2, CLIP, SAM, ULIP2
- **MLLMs:** Molmo (AllenAI), GPT-4o (OpenAI), PaliGemma (Google)
- **Build System:** Pixi (conda + PyPI), Docker
- **CLI Framework:** Click
- **Type Checker:** pyright (`basic` mode, 0 errors) — config in `pyrightconfig.json`
- **Key Libraries:** einops, scipy, scikit-learn, matplotlib, pandas, potpourri3d, open3d, pyrender, pyrr, fast-hdbscan

## Environment Setup

### Using Pixi (recommended)

```bash
# Install pixi (https://pixi.sh)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies (conda + PyPI, including CUDA extensions)
pixi install

# Set dataset paths
export KEYPOINTNET_DATASET_PATH="/path/to/KeypointNet/dataset"
export COLMAP_DATA_PATH="/path/to/colmap/data"
export HUMAN3M_DATA_PATH="/path/to/Human3M/data"

# Run commands via pixi
pixi run zerokey --help

# Dev environment (adds pyright, conda)
pixi install -e dev
```

### Using Docker

```bash
# Build image
docker build -t zerokey .

# Run with GPU access
docker run --gpus all zerokey zerokey eval --dataset keypointnet

# Or use docker compose (mounts dataset dirs and results)
export KEYPOINTNET_DATASET_PATH="/path/to/KeypointNet/dataset"
docker compose run --rm zerokey zerokey eval --dataset keypointnet
```

### Manual Setup (alternative)

```bash
conda create -n zerokey python=3.13
conda activate zerokey
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/gt4o4/pytorch3d.git"
pip install "git+https://github.com/gt4o4/KNN_CUDA.git"
pip install "git+https://github.com/gt4o4/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib"
pip install "git+https://github.com/openai/CLIP.git"
conda install -c conda-forge scikit-learn matplotlib open3d pyrender pyrr fast_hdbscan
pip install scipy pandas potpourri3d einops
pip install -e .
```

### CUDA Extension Forks

The packages `pytorch3d`, `KNN-CUDA`, and `pointnet2-ops` import `torch` in their `setup.py` at metadata extraction time. Upstream repos can't be resolved by pixi/uv because the solver needs to build metadata before installing anything, but `torch` isn't available yet (chicken-and-egg problem).

**Solution:** We use forks under `gt4o4/` that add `pyproject.toml` with `torch==2.10.0+cu130` in `build-system.requires`. This tells the PEP 517 build frontend to install torch in the isolated build environment before running setup.py:
- **`gt4o4/pytorch3d`** — fork of `facebookresearch/pytorch3d`
- **`gt4o4/KNN_CUDA`** — fork of `unlimblue/KNN_CUDA` (also fixed setup.py to extract version via regex instead of importing the package)
- **`gt4o4/Pointnet2_PyTorch`** — fork of `erikwijmans/Pointnet2_PyTorch` (pyproject.toml added to `pointnet2_ops_lib/` subdirectory)
- **`gt4o4/mmcv`** — fork of `open-mmlab/mmcv` (replaced `pkg_resources` with `importlib.metadata`, fixed `exec()+locals()` Python 3 bug)
- **`gt4o4/mmsegmentation`** — fork of `open-mmlab/mmsegmentation` (fixed same `exec()+locals()` bug, bumped MMCV_MAX to `2.3.0`)

All five are declared in `[tool.pixi.pypi-dependencies]` and resolved directly by `pixi install` — no post-install task needed.

## Running via CLI

All evaluation, baseline, metric, and visualization scripts are unified under a single CLI:

```bash
# Via pixi (recommended)
pixi run zerokey --help

# Or directly if environment is activated
zerokey --help
python -m zerokey --help
```

### Evaluation (Our Method)

```bash
# KeypointNet dataset evaluation (primary benchmark)
zerokey eval --dataset keypointnet

# Human3M dataset evaluation
zerokey eval --dataset human3m

# Real scene evaluation (COLMAP-based)
zerokey eval --dataset realscene

# Custom output directory and experiment name
zerokey eval --log-dir /path/to/output --expname MyExperiment
```

### Baselines

```bash
# PatchAlign3D evaluation
zerokey baseline patchalign3d

# PatchAlign3D with reference views
zerokey baseline patchalign3dref

# ULIP2 reference view evaluation
zerokey baseline ulip2ref

# BT3D benchmark
zerokey baseline bt3d

# Other baselines: gpt4o, redcircle, saliency, clip-dinoiser, paligemma, stable-keypoints
zerokey baseline --help
```

### Metrics

```bash
# IoU calculation
zerokey metric iou --expname ZeroKey

# Debug evaluation
zerokey metric debug

# Raw points evaluation
zerokey metric rawpts
```

### Visualization

```bash
# GPT-4o visualization demo
zerokey vis gpt4o

# Schelling point visualization
zerokey vis schelling

# Point describability
zerokey vis describe

# Demo pipeline
zerokey vis demo
```

### Data Preparation

```bash
# Sample shapes from KeypointNet splits
zerokey data sample --save-dir ./rendered --keypointnet-dir $KEYPOINT_DATASET_PATH

# Render sampled shapes (run after sample)
zerokey data render --save-dir ./rendered --keypointnet-dir $KEYPOINT_DATASET_PATH
```

### CLIP-DINOiser (separate module)

```bash
# CLIP-DINOiser evaluation (self-contained module)
torchrun clip_dinoiser/main_eval.py clip_dinoiser.yaml

# Multi-GPU CLIP-DINOiser
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 clip_dinoiser/main_eval.py clip_dinoiser.yaml
```

### Type Checking

```bash
# Run pyright (requires dev environment)
pixi run -e dev pyright

# Expected: 0 errors (warnings are acceptable)
```

The entire codebase (including `patchalign3d/`, `ULIP/`, `molmo/`, `data_creation/`, `clip_dinoiser/`, `unsupervised_keypoints/`) passes pyright with zero errors. Configuration is in `pyrightconfig.json` (`typeCheckingMode: "basic"`, Python 3.13). Third-party libraries without type stubs use `# type: ignore` annotations where needed.

## Architecture Overview

### Pipeline Architecture

The system follows a multi-stage pipeline:

```
3D Mesh
  → Multi-view Rendering (PyTorch3D)
  → Feature Extraction (DINO/CLIP/SAM)
  → Feature Backprojection to 3D
  → Point Localization (Molmo/GPT-4o)
  → Candidate Optimization (Quadratic Assignment)
  → Keypoint Detection + Naming
```

### Class Hierarchy

**Core Generator Pattern:**
- `RenderO3D` - Base class for PyTorch3D rendering
- `KPNetGenerator(RenderO3D, Generic[_IO, _M])` - Main pipeline orchestrator
  - Generic over `_IO` (I/O handler, bound to `KPNetIO`) and `_M` (multimodal model, unbound)
  - `self.multimodal: Optional[_M]` — lazily-initialized multimodal model instance
  - `self._multimodal` property — lazy-initializing, non-None accessor returning `_M`
  - Integrates `GPT4o` for keypoint naming
  - Default `Multimodal = Molmo` for point localization
  - Uses `KPNetIO` for dataset interaction
  - Key methods: `detect_kps()`, `backproject_kps()`, `aggregate_kps()`, `process_kp_list()`

**Dataset-Specific Variants:**
- `Human3MGenerator(KPNetGenerator)` - Human body keypoints with `Human3MIO`
- `RealSceneGenerator(KPNetGenerator)` - Scene keypoints with `RealSceneIO`

**PatchAlign3D Variants:**
- `PatchAlign3DGenerator(KPNetGenerator)` - PatchAlign3D baseline with patch feature extraction
- `PatchAlign3DZeroKeyGenerator(KPNetGenerator)` - Combines PatchAlign3D patch features with ZeroKey pipeline
  - Uses `FragmentsWithPts` frozen dataclass extending `Fragments` with pts/cameras/lights
  - Uses `SuperDict(UserDict)` to carry npz patch data and match results through the pipeline
  - Overrides `views_from_model()` to project mesh vertices into camera space per view
  - Overrides `init_kps_batch()` to extract patch features and match via CLIP text embeddings
  - Overrides `aggregate_kps()` with CLIP-enhanced HDBSCAN clustering:
    - Adjusts alpha weights by each point's own-view CLIP distance (percentile-normalized, pow(2) sharpened, 0.1x floor)
    - PCA-reduces per-point CLIP features [M, B, D] → [M*B, D] → PCA(3) → mean across views → [M, 3]
    - Clusters in 6D space [xyz, pca_feat * scale] with adjusted weights
    - Computes weighted centroids in 3D space only

### Key Design Patterns

1. **Lazy Initialization:** MLLMs are initialized on first use via the `_multimodal` property
   ```python
   # Property handles lazy init and None guard:
   molmo = cast(Molmo, self._multimodal)  # in base class methods
   # Subclasses can also access directly:
   self._multimodal.method(...)            # typed as _M
   ```

2. **Template Method:** Generator classes define pipeline structure in `main_loop()`

3. **Strategy Pattern:** Different I/O strategies via `KPIO` class attribute

## Directory Structure

```
.
├── pyproject.toml                             # Project config + pixi workspace
├── pixi.lock                                  # Locked dependency versions
├── Dockerfile                                 # CUDA 13.0 + pixi container
├── docker-compose.yml                         # GPU-enabled compose with dataset mounts
├── .dockerignore                              # Docker build exclusions
├── zerokey/                                   # Main package
│   ├── __main__.py                            # Entry point: python -m zerokey
│   ├── cli.py                                 # Click CLI definition
│   ├── commands/                              # CLI command groups
│   │   ├── eval.py                            # eval command (zerokey eval --dataset ...)
│   │   ├── baseline.py                        # baseline subcommands
│   │   ├── metric.py                          # metric subcommands (IoU, debug)
│   │   ├── vis.py                             # visualization subcommands
│   │   └── data.py                            # data preparation (sample, render)
│   ├── generators/                            # Pipeline generators (one per method)
│   │   ├── kpnet.py                           # KPNetGenerator - our main method
│   │   ├── human3m.py                         # Human3MGenerator
│   │   ├── realscene.py                       # RealSceneGenerator
│   │   ├── patchalign3d.py                    # PatchAlign3DGenerator (baseline)
│   │   ├── patchalign3dzerokey.py             # PatchAlign3DZeroKeyGenerator (baseline)
│   │   ├── patchalign3dref.py                 # PatchAlign3DRefGenerator (baseline)
│   │   ├── ulip2ref.py                        # ULIP2RefGenerator (baseline)
│   │   ├── bt3d.py                            # BT3D benchmark (baseline)
│   │   ├── gpt4o.py                           # GPT4oGenerator (baseline)
│   │   ├── paligemma.py                       # PaliGemmaGenerator (baseline)
│   │   ├── redcircle.py                       # RedCircleGenerator (baseline)
│   │   ├── saliency.py                        # SaliencyGenerator (baseline)
│   │   ├── clip_dinoiser.py                   # ClipDINOiser (baseline)
│   │   └── stable_keypoints.py                # StableKeypoints (baseline)
│   ├── io/                                    # I/O and evaluation classes
│   │   ├── kpnet.py                           # KPNetIO, KPNetEvaluator
│   │   ├── debug.py                           # KPNetEvalDebug
│   │   ├── rawpts.py                          # Raw points evaluator
│   │   └── schelling.py                       # SchellingIO
│   └── vis/                                   # Visualization scripts
│       ├── gpt4o.py                           # GPT-4o visualization
│       ├── describe.py                        # Point describability
│       ├── schelling.py                       # Schelling point visualization
│       └── demo.py                            # Demo pipeline
├── candidate_optimization.py                  # Quadratic assignment solver
├── data_creation/                             # Data generation & preprocessing
│   ├── common_data_utils.py                   # Shared data utilities
│   ├── pali_gemma.py                          # PaliGemma integration
│   ├── gemma3.py                              # Gemma3 integration
│   ├── keypointnet/                           # KeypointNet utilities
│   ├── scene/                                 # COLMAP integration
│   └── big_vision/                            # Vision model configs
├── feature_backprojection/                    # Feature extraction & projection
│   ├── backprojection.py                      # Multi-view feature extraction
│   ├── model_wrappers.py                      # DINO/SAM/CLIP wrappers
│   └── saliency_extractor.py                  # Saliency map extraction
├── kp_utils/                                  # Core utilities
│   ├── data/                                  # Dataset loaders (KeypointNet, Schelling)
│   ├── evaluation.py                          # IoU, geodesic metrics
│   ├── geometry.py                            # Geodesic distances, mesh ops
│   └── rendering.py                           # Renderer setup, viewpoints
├── patchalign3d/                              # Point-BERT patch alignment module
│   ├── models/                                # PointTransformer, PointTokenizer models
│   ├── data_utils/                            # ShapeNet, PartNet, Find3D dataloaders
│   ├── tools/                                 # CLI tools & evaluation scripts
│   │   ├── eval_cli.py                        # Unified PatchAlign3D evaluation CLI
│   │   ├── dump_matching_patch_features.py    # Patch feature extraction
│   │   ├── preprocess_faust_partnete.py       # FAUST/PartNetE preprocessing
│   │   └── seen_unseen_objaverse_general.py   # Objaverse seen/unseen splits
│   ├── inference/                             # Patch feature extraction & inference
│   ├── train_*.py                             # Training scripts (CLIP, DINOv2, part seg)
│   ├── test_partseg.py                        # Part segmentation testing
│   └── data -> ../../Point-BERT/data          # Symlink to Point-BERT data
├── clip_dinoiser/                             # Semantic segmentation module (self-contained)
├── molmo/                                     # Molmo implementation
├── ULIP/                                      # ULIP2 point cloud feature extraction
│   ├── models/                                # ULIP model architectures
│   │   ├── ULIP_models.py                     # Main ULIP2 model definitions
│   │   ├── pointbert/                         # PointBERT backbone & checkpoints
│   │   ├── pointmlp/                          # PointMLP backbone
│   │   ├── pointnet2/                         # PointNet++ backbone
│   │   └── pointnext/                         # PointNeXt backbone
│   └── data/                                  # Dataset configs and utilities
└── unsupervised_keypoints/                    # Experimental methods
├── tests/                                     # Unit tests (21 test files)
```

## Key Functions & Modules

### Rendering & Geometry
- `kp_utils/rendering.py::setup_renderer()` - Initialize PyTorch3D renderer
- `kp_utils/rendering.py::sample_view_points()` - Icosphere-based viewpoint sampling
- `kp_utils/geometry.py::pairwise_geodesic_distances()` - Geodesic distance computation
- `zerokey/rendering.py::views_from_model()` - Render mesh from multiple viewpoints, return images + fragments
- `zerokey/rendering.py::RenderO3D` - Unified 3D rendering interface
- `zerokey/rendering.py::camera_from_eye_at_up()` - Build FoVPerspectiveCameras from eye positions

### Feature Extraction
- `feature_backprojection/backprojection.py::features_from_views()` - Extract & aggregate features across views
- `feature_backprojection/backprojection.py::compute_kp_dists_features()` - Few-shot keypoint feature learning
- `feature_backprojection/model_wrappers.py` - Abstract wrappers for DINO/CLIP/SAM models

### Optimization
- `candidate_optimization.py::optimize_keypoint_candidates()` - Gradient-based quadratic assignment solver
  - Matches keypoint candidates using both feature similarity and geodesic distance preservation
  - Uses softmax selection matrix optimized with Adam
  - Parameters: `dist_alpha` (distance weight), `selection_beta` (sparsity reward)
- `candidate_optimization.py::optimize_linear()` - Hungarian algorithm baseline (features only)

### Evaluation
- `kp_utils/evaluation.py::eval_iou()` - Intersection-over-union for keypoint detection
- `kp_utils/evaluation.py::eval_det_cls()` - Per-class detection evaluation
- `kp_utils/evaluation.py::gen_geo_dists()` - Graph-based geodesic distance computation

### KPNetGenerator Pipeline (kpnet.py)

The core pipeline processes one keypoint prompt at a time via `process_kp_list()`:

1. **`init_kps_batch()`** — Optional batch pre-processing. Base returns empty dict. Subclasses (e.g. `PatchAlign3DZeroKeyGenerator`) extract patch features and pre-compute matches.

2. **`detect_kps()`** — Queries the MLLM (Molmo) per view to locate 2D keypoints. Returns `Dict[view_idx, point_coords]`.

3. **`backproject_kps()`** — Projects 2D detections into 3D using depth buffers and cameras. Draws colored circles at detected locations, maps colors to class IDs via `COLOR_MAP`. Produces `Pointclouds` with a **4-channel uint8 feature vector** per point:
   - `features[:, 0]` — view index (which rendered view)
   - `features[:, 1]` — point class ID (color-mapped index within view)
   - `features[:, 2]` — alpha/confidence weight
   - `features[:, 3]` — valid mask (1 if backprojection ray faces camera)

4. **`aggregate_kps()`** — Clusters 3D points via HDBSCAN into final keypoints. Reinterprets the first 2 uint8 feature channels as int16 for packed `(view, class)` identifiers. Uses per-class weight totals as a quality threshold: clusters must have total weight > 2× the max single-class weight to ensure multi-view support.

### Multimodal Models
- `zerokey/models/gpt4o.py::GPT4o` - OpenAI API integration for keypoint naming
- `zerokey/models/molmo.py::Molmo` - AllenAI Molmo for point localization
  - Default: `allenai/Molmo-7B-D-0924` (7B Dense)
  - Alternative: `allenai/MolmoE-1B-0924` (1B MoE, lightweight)
  - Alternative: `allenai/Molmo-72B-0924` (72B, highest quality)

## Configuration Patterns

Generators accept parameters through constructor:
```python
KPNetGenerator(
    log_dir=Path(),      # Output directory
    expname='ExpName',   # Experiment name
    res=512,             # Render resolution
    scale=2              # Upscaling factor for high-res rendering
)
```

Common configuration attributes:
- `self.views` - Viewpoint sampling (default: icosphere partition=3)
- `self.proj_radius` - Backprojection radius (default: 15)
- `self.dist` - Camera distance from object (default: 1)
- `self.device` - CUDA device

## Debugging & Visualization

Enable debug mode with environment variable:
```python
# In code: self.vis = debug_enabled()
# Controlled by environment variable or flag
```

When `self.vis` is enabled:
- Molmo visualizations show detected points on images
- Error messages are drawn directly on images
- Intermediate results are saved

## PatchAlign3D Module

The `patchalign3d/` directory contains a Point-BERT-based patch alignment system for 3D point cloud understanding. This module is used for experimental comparison and ablation studies.

**Key Components:**
- **Models:** PointTransformer variants with CLIP/DINOv2 alignment
  - `models/PointTransformer.py` - Base transformer architecture
  - `models/PointTransformer_patched.py` - Patch-level processing
  - `models/point_encoder.py` - Point cloud encoders
- **Training:** Multi-stage training scripts for different feature alignment strategies
  - `train_find3d_patch_clip_*.py` - CLIP-based patch training
  - `train_patch_clip_parts_zs_point.py` - Zero-shot part segmentation
  - `train_partseg*.py` - Part segmentation baselines
- **Data:** Custom dataloaders for ShapeNet, PartNet, Find3D datasets
  - `data_utils/find3d_dataset.py` - Find3D integration
  - `data_utils/ShapeNetDataLoader.py` - ShapeNet part segmentation
- **Evaluation:** Benchmark tools for various datasets
  - `tools/eval_cli.py` - Unified PatchAlign3D evaluation CLI
  - `tools/dump_matching_patch_features.py` - Patch feature extraction
  - `tools/preprocess_faust_partnete.py` - FAUST/PartNetE preprocessing
  - `tools/seen_unseen_objaverse_general.py` - Objaverse seen/unseen splits
  - `tools/VISUALISE*.ipynb` - Qualitative visualization notebooks

**Data Path:** The `patchalign3d/data` symlink points to `../../Point-BERT/data` which should contain the necessary datasets (ShapeNet, ModelNet, etc.)

## ULIP2 Module

The `ULIP/` directory contains the ULIP2 (Understanding Language-Image Pre-training for 3D) implementation for point cloud feature extraction. ULIP2 provides a unified representation learning framework that aligns 3D point cloud features with text and image features from pre-trained CLIP models.

**Key Components:**
- **Models:** Multiple point cloud backbone architectures
  - `models/ULIP_models.py` - Main ULIP2 model with CLIP integration (`ULIP2_PointBERT_Colored`, `ULIP2_WITH_OPENCLIP`)
  - `models/pointbert/` - PointBERT backbone (default for ULIP2)
  - `models/pointmlp/`, `models/pointnet2/`, `models/pointnext/` - Alternative backbones
- **Data:** Dataset utilities and configurations
  - `data/dataset_3d.py` - 3D dataset loading with `pc_normalize()` function

**Checkpoint Path:** The ULIP2 checkpoint is expected at `ULIP/models/pointbert/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`

**Integration:** The `zerokey/generators/ulip2ref.py` module uses ULIP2 for reference-based 3D keypoint detection:
- `ULIP2RefGenerator` - Generator class using ULIP2 for point cloud features
- `ULIP2RefIO` - I/O class for reference view sampling and feature matching
- Uses farthest point sampling (FPS) to resample point clouds to 10,000 points

## Important Notes

1. **CUDA Required:** PyTorch3D rendering requires GPU acceleration
2. **Memory Management:** MLLMs are lazily initialized to conserve VRAM
3. **Batch Processing:** Use `batch_size` parameter in `views_from_model()` to manage memory
4. **Dataset Dependencies:** Evaluation scripts require proper dataset paths via environment variables
5. **CLIP-DINOiser Independence:** The `clip_dinoiser/` module has its own `requirements.txt` and configs
6. **PatchAlign3D Data:** The `patchalign3d/` module requires Point-BERT datasets. Ensure the symlink `patchalign3d/data` correctly points to the Point-BERT data directory
7. **ULIP2 Checkpoint:** The ULIP2 module requires a pre-trained checkpoint at `ULIP/models/pointbert/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt`

## Common Development Workflows

### Adding a New MLLM Model
1. Create wrapper class in `zerokey/models/` (see `molmo.py`, `gpt4o.py` as examples)
2. Implement `generated_kps_points()` and `parse_points_str()` methods
3. Create a generator subclass with both type parameters:
   ```python
   class YourGenerator(KPNetGenerator[KPNetIO, YourModel]):
       Multimodal = YourModel
   ```

### Adding a New Dataset
1. Create I/O class inheriting from `KPNetIO` in `zerokey/io/`
2. Implement `loop_over_test_datasets()` generator method
3. Create generator class inheriting from `KPNetGenerator` in `zerokey/generators/`
4. Set `KPIO` class attribute: `YourGenerator.KPIO = YourIO`
5. Add CLI command in `zerokey/commands/eval.py` or `zerokey/commands/baseline.py`

### Modifying Feature Extraction
1. Add model wrapper in `feature_backprojection/model_wrappers.py`
2. Update `features_from_views()` in `feature_backprojection/backprojection.py`
3. Ensure compatibility with multi-view aggregation

## Testing

Unit tests are in the `tests/` directory (22 test files covering CLI, defaults, generators, I/O, rendering, geometry, and utilities):

```bash
# Run all tests
pixi run python -m pytest tests/ -v

# Run specific test
pixi run python -m pytest tests/test_cli.py -v
```

Additional tests exist in vendored submodules:
- `molmo/tests/` - Molmo model tests
- `data_creation/big_vision/` - Vision model tests (JAX-based)

## Git Workflow

Current branch: `release`

Recent development focuses on:
- Python 3.13, PyTorch 2.10.0+cu130, CUDA 13.0
- Pixi workspace with conda + PyPI dependency management (no post-install tasks)
- CUDA extensions (pytorch3d, KNN-CUDA, pointnet2-ops) resolved via gt4o4 forks with torch in build-system.requires
- Docker support (Dockerfile + docker-compose.yml with GPU)
- Restructured all scripts into `zerokey/` package with unified CLI (`python -m zerokey`)
- Switched default Molmo model to Molmo-7B-D-0924 (7B Dense variant)
- ULIP2 integration for point cloud feature extraction
- Reference view support for PatchAlign3D and ULIP2
- Multi-view feature alignment improvements
- Human3M dataset integration
- COLMAP-to-PyTorch3D camera conversion
- Merged and cleaned up `clip_dinoiser` and `patchalign3d+zerokey` branches
- Full pyright type checking enforced project-wide (0 errors, `basic` mode)
