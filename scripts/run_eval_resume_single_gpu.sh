#!/usr/bin/env bash
set -euo pipefail

# Single-GPU resumable ZeroKey eval runner.
# - Resume behavior: rerun with same --expname and --log-dir, completed meshes are skipped.
# - Saves:
#   - Raw Molmo 2D detections (JSON): Molmo2D_*_kps2d.json
#   - Final 3D predictions (PLY): *_keypts.ply
#
# Usage:
#   bash scripts/run_eval_resume_single_gpu.sh
#   GPU_ID=1 MAX_MESHES=10 EXPNAME=ZeroKeyResume bash scripts/run_eval_resume_single_gpu.sh

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-keypointnet}"
LOG_DIR="${LOG_DIR:-${ZEROKEY_LOG_DIR:-/data/taoye/zero-shot/results}}"
EXPNAME="${EXPNAME:-ZeroKeyResume}"
RES="${RES:-512}"
SCALE="${SCALE:-2}"
USE_TEXTURE="${USE_TEXTURE:-0}"   # 0 => --no-texture, 1 => --use-texture
MAX_MESHES="${MAX_MESHES:-0}"     # 0 => no limit

# Optional cache env (recommended for stable model loading)
if [[ -n "${HF_HOME:-}" ]]; then
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
fi

texture_flag="--no-texture"
if [[ "$USE_TEXTURE" == "1" ]]; then
  texture_flag="--use-texture"
fi

echo "[INFO] GPU_ID=$GPU_ID DATASET=$DATASET LOG_DIR=$LOG_DIR EXPNAME=$EXPNAME RES=$RES SCALE=$SCALE MAX_MESHES=$MAX_MESHES"
echo "[INFO] Resume mode is enabled by output existence checks in KPNetIO.check_if_complete."

set -x
CUDA_VISIBLE_DEVICES="$GPU_ID" pixi run zerokey eval \
  --dataset "$DATASET" \
  --log-dir "$LOG_DIR" \
  --expname "$EXPNAME" \
  "$texture_flag" \
  --res "$RES" \
  --scale "$SCALE" \
  --num-shards 1 \
  --shard-id 0 \
  --max-meshes "$MAX_MESHES"
set +x

echo "[INFO] Eval finished. Output root: $LOG_DIR/$EXPNAME"
