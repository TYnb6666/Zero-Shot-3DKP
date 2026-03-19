# ZeroKey: Zero-Shot 3D Keypoint Detection from Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2412.06292-b31b1b.svg)](https://arxiv.org/abs/2412.06292)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sites.google.com/view/zerokey)

Official implementation of **ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models**

**[Bingchen Gong](https://s2.hk/)<sup>1</sup>, [Diego Gomez](https://www.lix.polytechnique.fr/~gomez/)<sup>1</sup>, [Abdullah Hamdi](https://abdullahamdi.com/)<sup>2</sup>, [Abdelrahman Eldesokey](https://abdo-eldesokey.github.io/)<sup>3</sup>, [Ahmed Abdelreheem](https://samir55.github.io/)<sup>3</sup>, [Peter Wonka](https://peterwonka.net/)<sup>3</sup>, [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)<sup>1</sup>**

<sup>1</sup>École Polytechnique, <sup>2</sup>University of Oxford, <sup>3</sup>KAUST

---

## Overview

We propose a novel zero-shot approach for keypoint detection on 3D shapes. Point-level reasoning on visual data is challenging as it requires precise localization capability, posing problems even for powerful models like DINO or CLIP. Traditional methods for 3D keypoint detection rely heavily on annotated 3D datasets and extensive supervised training, limiting their scalability and applicability to new categories or domains. In contrast, our method utilizes the rich knowledge embedded within Multi-Modal Large Language Models (MLLMs). Specifically, we demonstrate, for the first time, that pixel-level annotations used to train recent MLLMs can be exploited for both extracting and naming salient keypoints on 3D models without any ground truth labels or supervision. Experimental evaluations demonstrate that our approach achieves competitive performance on standard benchmarks compared to supervised methods, despite not requiring any 3D keypoint annotations during training. Our results highlight the potential of integrating language models for localized 3D shape understanding.

<p align="center">
  <img src="docs/teaser3.png" alt="Zero-shot 3D Keypoint Detection" width="100%">
</p>

**Zero-shot 3D Keypoint Detection.** Without any ground truth labels or supervised training, our method leverages the point-level reasoning embedded within MLLMs to extract and name salient keypoints on 3D models. The figure illustrates how our approach achieves competitive performance compared to CLIP-DINOiser baselines, highlighting the potential of integrating language models with vision tasks for enhanced 3D shape understanding.

---

## Installation

### Prerequisites
- Python 3.13+
- CUDA 13.0+ (required for PyTorch3D)
- [Pixi](https://pixi.sh) (recommended) or Conda/Miniconda

### Quick Start with Pixi (recommended)

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies (conda + PyPI + CUDA extensions)
pixi install

# Verify installation
pixi run zerokey --help
```

### Type Checking

The entire codebase passes [pyright](https://github.com/microsoft/pyright) with zero errors under `basic` type checking mode:

```bash
# Run type checker (requires dev environment)
pixi run -e dev pyright
```

Configuration is in `pyrightconfig.json` (Python 3.13, `typeCheckingMode: "basic"`).

---

## Configuration

All configurable paths are centralized in `zerokey/_defaults.py` and read from environment variables with sensible defaults. You can configure them in two ways:

1. **Edit `zerokey/_defaults.py` directly** — change the fallback values in `os.environ.get()` calls
2. **Set environment variables in `pyproject.toml`** — add entries under `[tool.pixi.activation.env]` so they are automatically set when the pixi environment is activated:
   ```toml
   [tool.pixi.activation.env]
   KEYPOINT_DATASET_PATH = "/path/to/KeypointNet/dataset"
   ```

| Variable | Default | Description |
|---|---|---|
| `KEYPOINT_DATASET_PATH` | `keypointnet` | KeypointNet dataset root |
| `COLMAP_DATA_PATH` | *(empty)* | COLMAP scene data (real-scene eval) |
| `HUMAN3M_DATA_PATH` | *(empty)* | Human3.6M scans (human body eval) |
| `ZEROKEY_LOG_DIR` | `~/zerokey-results` | Output directory for results |
| `SHAPENET_PART_ROOT` | `shapenetcore_partanno_...` | ShapeNet part segmentation |
| `FIND3D_DATA_ROOT` | `Find3D/data_root` | Find3D dataset root |
| `PARTNET_ROOT` | `PartNet` | PartNet dataset root |
| `OBJAVERSE_GENERAL_ROOT` | `Find3D/obja_benchmark/...` | Objaverse-General-Find3D root |
| `POINTBERT_CKPT` | `model_ckpt/2stagemodel.pt` | Point-BERT checkpoint |
| `POINTBERT_SEG_EXPERIMENTS` | `pointbert_seg_experiments` | PatchAlign3D segmentation experiment dir |
| `OPENAI_API_KEY` | *(none)* | For GPT-4o baseline (read by OpenAI SDK) |

### Datasets

1. **KeypointNet** (Primary Benchmark)
   - Download: [KeypointNet Repository](https://github.com/qq456cvb/KeypointNet)
   - Set `KEYPOINT_DATASET_PATH` in your environment or `.env`

2. **Human3M** (Optional - for human body evaluation)
   - Set `HUMAN3M_DATA_PATH`

3. **COLMAP Data** (Optional - for real scene evaluation)
   - Set `COLMAP_DATA_PATH`

4. **Point-BERT Data** (Optional - for PatchAlign3D module)
   - Required for `patchalign3d/` experiments
   - Set `POINTBERT_CKPT` to the checkpoint path

---

## Usage

All evaluation, baseline, metric, and visualization scripts are unified under a single CLI:

```bash
# Via pixi (recommended)
pixi run zerokey --help

# Or directly if environment is activated
zerokey --help
```

### Evaluation (Our Method)

```bash
# KeypointNet dataset evaluation (primary benchmark)
zerokey eval --dataset keypointnet

# Human3M dataset evaluation
zerokey eval --dataset human3m

# Real scene evaluation (COLMAP-based)
zerokey eval --dataset realscene
```

### Baselines

```bash
# PatchAlign3D (unified: --mode selects variant)
zerokey baseline patchalign3d                    # base patch matching
zerokey baseline patchalign3d --mode zerokey      # hybrid MLLM + patch
zerokey baseline patchalign3d --mode ref           # reference-view

# Other baselines
zerokey baseline ulip2ref
zerokey baseline bt3d
zerokey baseline gpt4o
zerokey baseline clip-dinoiser
zerokey baseline redcircle
zerokey baseline saliency
zerokey baseline stable-keypoints
zerokey baseline paligemma

# List all available baselines
zerokey baseline --help
```

### Metrics

```bash
# IoU calculation
zerokey metric iou --expname ZeroKey

# Debug and raw points evaluation
zerokey metric debug --expname ZeroKey
zerokey metric rawpts --expname ZeroKey

# Schelling dataset evaluation
zerokey metric schelling --expname ZeroKey
```

### Visualization

```bash
# Interactive annotation tools (open image file windows)
zerokey vis gpt4o <image_path>
zerokey vis demo <image_path>

# Dataset annotation loops (require corresponding dataset)
zerokey vis schelling
zerokey vis describe
```

### Data Preparation

```bash
# Step 1: Sample shapes from KeypointNet splits (creates train_shapes.csv)
zerokey data sample --save-dir ./rendered --keypointnet-dir $KEYPOINT_DATASET_PATH

# Step 2: Render sampled shapes (requires step 1 to have been run first)
zerokey data render --save-dir ./rendered --keypointnet-dir $KEYPOINT_DATASET_PATH
```

## Project Structure

```
.
├── zerokey/                        # Main package (CLI + pipeline)
│   ├── cli.py                      # Click CLI definition
│   ├── _defaults.py                # Centralized env-var-backed path defaults
│   ├── _detection.py               # KeypointDetectionMixin (shared detection methods)
│   ├── rendering.py                # RenderO3D - PyTorch3D rendering base
│   ├── candidate_optimization.py   # Quadratic assignment solver
│   ├── commands/                   # CLI command groups
│   │   ├── eval.py                 # eval command (zerokey eval --dataset ...)
│   │   ├── baseline.py             # baseline subcommands
│   │   ├── metric.py               # metric subcommands (IoU, debug, rawpts, schelling)
│   │   ├── vis.py                  # visualization subcommands
│   │   └── data.py                 # data preparation (sample, render)
│   ├── models/                     # Multimodal model wrappers
│   │   ├── molmo.py                # Molmo model integration
│   │   ├── gpt4o.py                # GPT-4o API integration
│   │   └── red_circle.py           # Red-circle prompting wrapper
│   ├── generators/                 # Pipeline generators (one per method)
│   │   ├── kpnet.py                # KPNetGenerator[_IO, _M] - our main method
│   │   ├── human3m.py              # Human3MGenerator
│   │   ├── realscene.py            # RealSceneGenerator
│   │   ├── patchalign3d.py         # PatchAlign3DGenerator (--mode patch|zerokey|ref)
│   │   ├── ulip2ref.py             # ULIP2RefGenerator (baseline)
│   │   ├── gpt4o.py                # GPT4oGenerator (baseline)
│   │   ├── paligemma.py            # PaliGemmaGenerator (baseline)
│   │   ├── clip_dinoiser.py        # ClipDINOiserGenerator (baseline)
│   │   ├── redcircle.py            # RedCircleGenerator (baseline)
│   │   ├── saliency.py             # SaliencyGenerator (baseline)
│   │   ├── stable_keypoints.py     # StableKeypoints (baseline)
│   │   └── bt3d.py                 # BT3D benchmark (baseline)
│   ├── io/                         # I/O and evaluation classes
│   │   ├── kpnet.py                # KPNetIO, KPNetEvaluator, RefIO
│   │   ├── debug.py                # KPNetEvalDebug
│   │   ├── rawpts.py               # Raw points evaluator
│   │   └── schelling.py            # SchellingIO
│   └── vis/                        # Visualization scripts
│       ├── base.py                 # VisGeneratorBase
│       ├── gpt4o.py                # GPT-4o annotation
│       ├── demo.py                 # Demo pipeline
│       ├── describe.py             # Point describability
│       └── schelling.py            # Schelling point visualization
│
├── data_creation/                  # Data generation & preprocessing
│   ├── keypointnet/                # KeypointNet sampling & rendering tools
│   ├── big_vision/                 # PaliGemma / big_vision integration
│   ├── mvimgnet/                   # MVImgNet data preparation
│   ├── scene/                      # COLMAP / Gaussian Splatting utilities
│   ├── pali_gemma.py               # PaliGemma model integration
│   ├── gemma3.py                   # Gemma3 model integration
│   └── common_data_utils.py        # Shared data utilities
│
├── feature_backprojection/         # Feature extraction & projection
│   ├── backprojection.py           # Multi-view feature extraction
│   ├── model_wrappers.py           # DINO/SAM/CLIP wrappers
│   └── saliency_extractor.py       # Saliency map extraction
│
├── kp_utils/                       # Core utilities
│   ├── data/                       # Dataset loaders (KeypointNet, Schelling)
│   ├── evaluation.py               # IoU, geodesic metrics
│   ├── geometry.py                 # Geodesic distances, mesh ops
│   └── rendering.py                # Renderer setup, viewpoints
│
├── patchalign3d/                   # Point-BERT patch alignment
│   ├── models/                     # PointTransformer, PointTokenizer
│   ├── data_utils/                 # ShapeNet, PartNet, Find3D dataloaders
│   ├── inference/                  # PatchExplorer, patch feature extraction
│   │   └── explore_pc_patches.py   # PatchExplorer (inherits Molmo)
│   └── tools/                      # CLI tools & evaluation scripts
│       ├── eval_cli.py             # Unified PatchAlign3D evaluation CLI
│       ├── dump_matching_patch_features.py  # Patch feature extraction
│       ├── preprocess_faust_partnete.py     # FAUST/PartNetE preprocessing
│       └── seen_unseen_objaverse_general.py # Objaverse seen/unseen splits
│
├── molmo/                          # Molmo model implementation
├── ULIP/                           # ULIP2 point cloud feature extraction
├── clip_dinoiser/                  # CLIP-DINOiser semantic segmentation
├── unsupervised_keypoints/         # Experimental unsupervised methods
├── tests/                          # Unit tests (23 test files)
├── pyproject.toml                  # Project config + pixi workspace
├── pyrightconfig.json              # Type checking config (basic mode)
├── pixi.lock                       # Locked dependency versions
├── Dockerfile                      # CUDA 13.0 + pixi container
└── docker-compose.yml              # GPU-enabled compose with dataset mounts
```

---

## Pipeline Architecture

The system follows a multi-stage pipeline:

```
3D Mesh
  ↓
Multi-view Rendering (PyTorch3D)
  ↓
Feature Extraction (DINOv2/CLIP/SAM)
  ↓
Feature Backprojection to 3D
  ↓
Point Localization (Molmo/GPT-4o)
  ↓
Candidate Optimization (Quadratic Assignment)
  ↓
Keypoint Detection + Semantic Naming
```

### Class Hierarchy

```
RenderO3D                          # Base rendering (PyTorch3D)
└── KPNetGenerator[_IO, _M]       # Main ZeroKey pipeline orchestrator (Generic)
    ├── Human3MGenerator           # Human body keypoints (Human3MIO, Molmo)
    ├── RealSceneGenerator         # Real scene keypoints (RealSceneIO, Molmo)
    ├── PatchAlign3DGenerator      # PatchAlign3D (--mode: patch|zerokey|ref)
    │   └── ULIP2RefGenerator      # ULIP2 reference view baseline
    ├── GPT4oGenerator             # GPT-4o localization baseline
    ├── PaliGemmaGenerator         # PaliGemma baseline
    ├── RedCircleGenerator         # Red circle prompting baseline
    ├── SaliencyGenerator          # Saliency-based (DINOv2) baseline
    ├── ClipDINOiserGenerator      # CLIP-DINOiser baseline
    └── StableKeypoints            # Unsupervised keypoints baseline

Molmo                              # MLLM for point localization
└── PatchExplorer                  # Patch-level 3D exploration (inherits Molmo)
```

### Key Components

1. **Rendering & Geometry**
   - PyTorch3D-based multi-view rendering
   - Icosphere-based viewpoint sampling
   - Geodesic distance computation

2. **Feature Extraction**
   - DINOv2, CLIP, SAM model wrappers
   - Multi-view feature aggregation
   - Saliency map extraction

3. **MLLM Integration**
   - Molmo for pixel-level point localization
   - GPT-4o for semantic keypoint naming
   - Lazy initialization for memory efficiency

4. **Optimization**
   - Quadratic assignment problem solver
   - Feature similarity + geodesic distance preservation
   - Hungarian algorithm baseline

5. **Backprojection Feature Format**
   - Per-point features are 4-channel uint8 values: `[view_idx, class_id, alpha, valid]`
   - `view_idx`: which rendered view the point came from
   - `class_id`: semantic class encoded as a color index
   - `alpha`: confidence weight
   - `valid`: mask flag (1 = ray faces camera)
   - Two uint8 channels are packed into int16 via `.view(torch.int16)` for efficient (view, class) grouping

---

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Gong_2025_ICCV,
    author    = {Gong, Bingchen and Gomez, Diego and Hamdi, Abdullah and Eldesokey, Abdelrahman and Abdelreheem, Ahmed and Wonka, Peter and Ovsjanikov, Maks},
    title     = {ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {22089-22099}
}
```

---

## License

This project is licensed under the terms specified in the repository.

## Acknowledgments

We thank the authors of PyTorch3D, DINOv2, CLIP, SAM, Molmo, and Point-BERT for their excellent work and open-source contributions.
