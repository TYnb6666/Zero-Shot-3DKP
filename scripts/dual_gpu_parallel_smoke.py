"""Small dual-GPU smoke/throughput test.

This script verifies that 2 GPUs can run workloads concurrently and reports
serial-vs-parallel timing for a synthetic CUDA workload.

Usage:
  pixi run python scripts/dual_gpu_parallel_smoke.py
  pixi run python scripts/dual_gpu_parallel_smoke.py --warmup 10 --steps 80
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, asdict

import torch


@dataclass
class WorkerResult:
    gpu_id: int
    ok: bool
    elapsed_s: float
    device_name: str
    peak_mem_mb: float
    error: str = ""


def gpu_workload(gpu_id: int, size: int, warmup: int, steps: int, dtype: str) -> WorkerResult:
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

        dtypes = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        t = dtypes[dtype]

        a = torch.randn((size, size), device=device, dtype=t)
        b = torch.randn((size, size), device=device, dtype=t)

        for _ in range(warmup):
            _ = a @ b
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(steps):
            _ = a @ b
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return WorkerResult(
            gpu_id=gpu_id,
            ok=True,
            elapsed_s=elapsed,
            device_name=torch.cuda.get_device_name(device),
            peak_mem_mb=peak,
        )
    except Exception as e:  # keep process-local traceback isolated to this worker
        return WorkerResult(
            gpu_id=gpu_id,
            ok=False,
            elapsed_s=0.0,
            device_name="unknown",
            peak_mem_mb=0.0,
            error=repr(e),
        )


def run_parallel(gpus: list[int], size: int, warmup: int, steps: int, dtype: str) -> tuple[float, list[WorkerResult]]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpus)) as pool:
        t0 = time.perf_counter()
        results = pool.starmap(
            gpu_workload,
            [(gid, size, warmup, steps, dtype) for gid in gpus],
        )
        wall = time.perf_counter() - t0
    return wall, results


def run_serial(gpus: list[int], size: int, warmup: int, steps: int, dtype: str) -> tuple[float, list[WorkerResult]]:
    t0 = time.perf_counter()
    results = [gpu_workload(gid, size, warmup, steps, dtype) for gid in gpus]
    wall = time.perf_counter() - t0
    return wall, results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-GPU parallel smoke/throughput test")
    p.add_argument("--gpus", default="0,1", help="Comma-separated CUDA ids (default: 0,1)")
    p.add_argument("--size", type=int, default=4096, help="Square GEMM size (default: 4096)")
    p.add_argument("--warmup", type=int, default=8, help="Warmup iterations per GPU (default: 8)")
    p.add_argument("--steps", type=int, default=40, help="Measured iterations per GPU (default: 40)")
    p.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Compute dtype (default: bf16)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 2

    visible = torch.cuda.device_count()
    requested = [int(x) for x in args.gpus.split(",") if x.strip()]
    if len(requested) < 2:
        print("Please provide at least 2 GPU ids via --gpus.")
        return 2
    if any(g < 0 or g >= visible for g in requested):
        print(f"Invalid --gpus {requested}; visible cuda devices = {visible}")
        return 2

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"Visible GPUs: {visible}")
    print(f"Requested GPUs: {requested}")
    print(f"Config: size={args.size}, warmup={args.warmup}, steps={args.steps}, dtype={args.dtype}")

    serial_wall, serial_results = run_serial(requested, args.size, args.warmup, args.steps, args.dtype)
    parallel_wall, parallel_results = run_parallel(requested, args.size, args.warmup, args.steps, args.dtype)

    print("\n[Serial results]")
    for r in serial_results:
        print(asdict(r))
    print(f"serial_wall_s={serial_wall:.3f}")

    print("\n[Parallel results]")
    for r in parallel_results:
        print(asdict(r))
    print(f"parallel_wall_s={parallel_wall:.3f}")

    ok = all(r.ok for r in parallel_results)
    speedup = serial_wall / parallel_wall if parallel_wall > 0 else 0.0
    print(f"\nparallel_ok={ok}, speedup_vs_serial={speedup:.3f}x")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

