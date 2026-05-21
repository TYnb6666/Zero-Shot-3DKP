#!/usr/bin/env python3
"""Quick checker for single-GPU resumable eval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--expname", type=str, required=True)
    args = parser.parse_args()

    root = args.log_dir / args.expname
    if not root.exists():
        print(f"[ERR] output root not found: {root}")
        return 2

    kps2d = list(root.glob("*/*/Molmo2D_*_kps2d.json"))
    keypts = list(root.glob("*/*/*_keypts.ply"))

    print(f"[INFO] root: {root}")
    print(f"[INFO] Molmo 2D JSON files: {len(kps2d)}")
    print(f"[INFO] Final keypoint PLY files: {len(keypts)}")
    if kps2d:
        print(f"[INFO] sample 2D JSON: {kps2d[0]}")
    if keypts:
        print(f"[INFO] sample keypts PLY: {keypts[0]}")

    if not kps2d:
        print("[WARN] no 2D JSON found")
    if not keypts:
        print("[WARN] no final keypoint PLY found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
