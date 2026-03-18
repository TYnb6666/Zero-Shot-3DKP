"""Centralized path defaults — one place to define all env-var-backed paths.

External env vars consumed by third-party libraries (not defined here):
  OPENAI_API_KEY, OPENAI_ORG_ID  — OpenAI SDK
  HF_HOME                        — HuggingFace Hub
  MOLMO_DATA_DIR                  — Molmo data loader
"""

import os
from pathlib import Path

# ZeroKey
DEFAULT_LOG_DIR = Path(
    os.environ.get("ZEROKEY_LOG_DIR", str(Path.home() / "zerokey-results"))
)

# KeypointNet
KEYPOINT_DATASET_PATH = os.environ.get("KEYPOINT_DATASET_PATH", "keypointnet")

# RealScene / Human3M
COLMAP_DATA_PATH = os.environ.get("COLMAP_DATA_PATH", "")
HUMAN3M_DATA_PATH = os.environ.get("HUMAN3M_DATA_PATH", "")

# PatchAlign3D — datasets
SHAPENET_PART_ROOT = os.environ.get(
    "SHAPENET_PART_ROOT",
    "shapenetcore_partanno_segmentation_benchmark_v0_normal",
)
FIND3D_DATA_ROOT = os.environ.get("FIND3D_DATA_ROOT", "Find3D/data_root")
PARTNET_ROOT = os.environ.get("PARTNET_ROOT", "PartNet")
OBJAVERSE_GENERAL_ROOT = os.environ.get(
    "OBJAVERSE_GENERAL_ROOT",
    "Find3D/obja_benchmark/Objaverse-General-Find3D/objaverse-general",
)

# PatchAlign3D — checkpoints / experiments
POINTBERT_CKPT = os.environ.get("POINTBERT_CKPT", "model_ckpt/2stagemodel.pt")
POINTBERT_SEG_EXPERIMENTS = os.environ.get(
    "POINTBERT_SEG_EXPERIMENTS", "pointbert_seg_experiments"
)
