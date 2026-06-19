#!/usr/bin/env python3
"""Audit Molmo 2D predictions against visible KeypointNet GT projections.

This utility reads saved ``Molmo2D_*_kps2d.json`` files, projects KeypointNet
3D ground-truth keypoints into the same rendered view coordinates, filters out
views where the target GT keypoints are occluded using trimesh ray casting, and
reports best-case per-mesh/per-class Molmo view accuracy statistics.

It intentionally evaluates **Molmo2D vs visible GT**, not final ZeroKey 3D
predictions.  Occluded-view Molmo predictions are counted separately and are not
included in error statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Heavy project dependencies (torch / pytorch3d / trimesh) are imported lazily
# in run_audit() so --help works in lightweight environments.


DEFAULT_CLASSES = ("airplane", "chair", "table")


@dataclass(frozen=True)
class Detection:
    index: int
    x_pct: float
    y_pct: float


@dataclass(frozen=True)
class VisibleGT:
    semantic_id: int
    x: float
    y: float
    z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit saved Molmo 2D detections against visible GT projections and select best views per mesh.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/data/taoye/zero-shot/results/ZeroKeyResume"),
        help="Experiment directory containing class/mesh/Molmo2D_*_kps2d.json files.",
    )
    parser.add_argument(
        "--keypointnet-dir",
        type=Path,
        default=Path(os.environ.get("KEYPOINT_DATASET_PATH", "/data/taoye/zero-shot/data/keypointnet")),
        help="KeypointNet dataset root containing annotations/ and ShapeNetCore.v2.ply/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/taoye/zero-shot/results/audit_molmo2d_visible_gt_bestview"),
        help="Directory where CSV/JSON audit reports will be written.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Class names to audit.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=1024,
        help="Screen-space resolution used for projection comparison, usually eval res*scale.",
    )
    parser.add_argument(
        "--view-radius",
        type=float,
        default=1.0,
        help="Camera view radius used by KPNetGenerator.",
    )
    parser.add_argument(
        "--view-partition",
        type=int,
        default=3,
        help="Camera view partition used by KPNetGenerator.",
    )
    parser.add_argument(
        "--visibility-threshold-ratio",
        type=float,
        default=0.015,
        help="Ray-hit distance tolerance as a fraction of trimesh bbox diagonal.",
    )
    parser.add_argument(
        "--min-visible-detections-per-view",
        type=int,
        default=1,
        help="Minimum visible/evaluable detections for a mesh view to be eligible as that mesh's best view.",
    )
    parser.add_argument(
        "--max-meshes-per-class",
        type=int,
        default=0,
        help="Debug limit per class; 0 means all meshes.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for PyTorch3D projection. Use auto to select cuda when available.",
    )
    parser.add_argument(
        "--mesh-subdir",
        default="ShapeNetCore.v2.ply",
        help="Mesh subdirectory under --keypointnet-dir.",
    )
    parser.add_argument(
        "--ray-engine",
        choices=("zbuf", "triangle", "pyembree"),
        default="zbuf",
        help=(
            "Visibility backend. Default zbuf uses PyTorch3D rasterized depth and avoids "
            "pyembree segfaults and triangle-backend rtree requirements."
        ),
    )
    parser.add_argument(
        "--zbuf-depth-threshold",
        type=float,
        default=0.01,
        help="Absolute screen/NDC z tolerance for z-buffer visibility checks.",
    )
    parser.add_argument(
        "--zbuf-window-radius",
        type=int,
        default=1,
        help="Pixel window radius around projected GT used for z-buffer visibility checks.",
    )
    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=4,
        help="Number of views to rasterize at once when --ray-engine=zbuf.",
    )
    return parser.parse_args()


def finite_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "variance": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "variance": float(np.var(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def as_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_csv_value(row.get(key, "")) for key in fieldnames})


def flatten_molmo_detections(view_payload: Any) -> list[Detection]:
    detections: list[Detection] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict) and "x" in value and "y" in value:
            try:
                detections.append(Detection(len(detections), float(value["x"]), float(value["y"])))
            except (TypeError, ValueError):
                return
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(view_payload)
    return detections


def molmo_to_pixel(det: Detection, res: int) -> tuple[float, float]:
    return det.x_pct / 100.0 * res, det.y_pct / 100.0 * res


def mesh_path_for(keypointnet_dir: Path, mesh_subdir: str, synset: str, mesh_id: str) -> Path:
    return keypointnet_dir / mesh_subdir / synset / f"{mesh_id}.ply"


def get_camera_batch(mesh: Any, views: np.ndarray, device: Any) -> Any:
    target = mesh.verts_packed().mean(dim=0, keepdim=True)
    eyes = torch.as_tensor(views, dtype=target.dtype, device=device) + target
    return camera_from_eye_at_up(eyes, target, device=device)


def zbuf_visible_mask(
    screen: np.ndarray,
    zbuf_view: Any,
    z_threshold: float,
    window_radius: int,
) -> list[bool]:
    if screen.size == 0:
        return []
    zbuf_np = zbuf_view.detach().cpu().numpy() if hasattr(zbuf_view, "detach") else np.asarray(zbuf_view)
    height, width = zbuf_np.shape
    visible: list[bool] = []
    radius = max(0, int(window_radius))
    for x, y, z in screen:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if not (z > 0 and 0 <= xi < width and 0 <= yi < height):
            visible.append(False)
            continue
        x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
        patch = zbuf_np[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            visible.append(False)
            continue
        visible.append(float(np.min(np.abs(patch - float(z)))) <= z_threshold)
    return visible


def ray_visible_mask(
    ray_intersector: Any,
    mesh_bounds: np.ndarray,
    camera_center: np.ndarray,
    gt_points: Sequence[Sequence[float]],
    threshold_ratio: float,
) -> list[bool]:
    if not gt_points:
        return []
    gt_np = np.asarray(gt_points, dtype=np.float64)
    centers = np.repeat(camera_center.reshape(1, 3), gt_np.shape[0], axis=0)
    dirs = gt_np - centers
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12)

    try:
        locations, index_ray, _index_tri = ray_intersector.intersects_location(
            ray_origins=centers,
            ray_directions=dirs,
            multiple_hits=False,
        )
    except Exception as exc:
        print(f"[WARN] trimesh ray casting failed: {exc}", file=sys.stderr)
        return [False] * len(gt_points)

    gt_dists = np.linalg.norm(gt_np - centers, axis=-1)
    bbox_diag = float(np.linalg.norm(mesh_bounds[1] - mesh_bounds[0]))
    threshold = bbox_diag * threshold_ratio

    visible = [False] * len(gt_points)
    for idx in range(len(gt_points)):
        ray_hits = index_ray == idx
        if not np.any(ray_hits):
            continue
        hit_dist = float(np.linalg.norm(locations[ray_hits][0] - centers[idx]))
        visible[idx] = abs(hit_dist - float(gt_dists[idx])) <= threshold
    return visible


def semantic_ids_from_payload(payload: dict[str, Any]) -> list[int]:
    ids = payload.get("semantic_ids", [])
    if not isinstance(ids, list):
        return []
    out: list[int] = []
    for semantic_id in ids:
        try:
            out.append(int(semantic_id))
        except (TypeError, ValueError):
            continue
    return out


def collect_gt_points(gt_kps: Iterable[dict[str, Any]], semantic_ids: set[int]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for kp in gt_kps:
        try:
            semantic_id = int(kp.get("semantic_id"))
        except (TypeError, ValueError):
            continue
        if semantic_id not in semantic_ids:
            continue
        xyz = kp.get("xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            continue
        points.append({"semantic_id": semantic_id, "xyz": [float(x) for x in xyz]})
    return points


def project_visible_gt(
    gt_points: list[dict[str, Any]],
    camera: Any,
    camera_center: np.ndarray,
    ray_intersector: Any | None,
    mesh_bounds: np.ndarray | None,
    zbuf_view: Any | None,
    ray_engine: str,
    res: int,
    visibility_threshold_ratio: float,
    zbuf_depth_threshold: float,
    zbuf_window_radius: int,
    device: Any,
) -> list[VisibleGT]:
    if not gt_points:
        return []
    xyz = [point["xyz"] for point in gt_points]
    gt_tensor = torch.tensor(xyz, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        screen = camera.transform_points_screen(gt_tensor, image_size=(res, res))[0].detach().cpu().numpy()
    if ray_engine == "zbuf":
        if zbuf_view is None:
            ray_visible = [False] * len(gt_points)
        else:
            ray_visible = zbuf_visible_mask(screen, zbuf_view, zbuf_depth_threshold, zbuf_window_radius)
    else:
        if ray_intersector is None or mesh_bounds is None:
            ray_visible = [False] * len(gt_points)
        else:
            ray_visible = ray_visible_mask(ray_intersector, mesh_bounds, camera_center, xyz, visibility_threshold_ratio)

    visible: list[VisibleGT] = []
    for point, coords, occl_free in zip(gt_points, screen, ray_visible):
        x, y, z = map(float, coords.tolist())
        if z > 0 and 0 <= x < res and 0 <= y < res and occl_free:
            visible.append(VisibleGT(int(point["semantic_id"]), x, y, z))
    return visible


def add_error_row(
    rows: list[dict[str, Any]],
    *,
    class_title: str,
    mesh_id: str,
    prompt: str,
    semantic_ids: list[int],
    view_idx: int,
    detection: Detection,
    gt: VisibleGT,
    match_mode: str,
    res: int,
    visibility_threshold_ratio: float,
) -> None:
    molmo_px, molmo_py = molmo_to_pixel(detection, res)
    error = float(math.hypot(molmo_px - gt.x, molmo_py - gt.y))
    rows.append({
        "class_title": class_title,
        "mesh_id": mesh_id,
        "prompt": prompt,
        "semantic_ids": ",".join(map(str, semantic_ids)),
        "view_idx": int(view_idx),
        "detection_index": int(detection.index),
        "match_mode": match_mode,
        "matched_semantic_id": int(gt.semantic_id),
        "molmo_px": molmo_px,
        "molmo_py": molmo_py,
        "gt_px": float(gt.x),
        "gt_py": float(gt.y),
        "gt_pz": float(gt.z),
        "error_px": error,
        "visibility_threshold_ratio": float(visibility_threshold_ratio),
    })


def render_zbuf_views(renderer: Any, mesh: Any, cameras: Any, batch_size: int) -> Any:
    zbuf_parts = []
    num_views = len(cameras)
    batch = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, num_views, batch):
            stop = min(num_views, start + batch)
            cam_batch = cameras[torch.arange(start, stop, device=cameras.device)]
            fragments = renderer.rasterizer(mesh.extend(stop - start), cameras=cam_batch)
            zbuf_parts.append(fragments.zbuf[..., 0].detach().cpu())
    return torch.cat(zbuf_parts, dim=0)


def evaluate_detection_matches(
    rows: list[dict[str, Any]],
    *,
    class_title: str,
    mesh_id: str,
    prompt: str,
    semantic_ids: list[int],
    view_idx: int,
    detections: list[Detection],
    visible_gt: list[VisibleGT],
    res: int,
    visibility_threshold_ratio: float,
) -> dict[str, int]:
    counts = Counter()
    if not detections:
        return counts
    if not visible_gt:
        counts["occluded_gt_but_molmo_predicted"] += len(detections)
        return counts

    # group_oracle: lower-bound/best-case nearest visible GT within the prompt group.
    for detection in detections:
        molmo_px, molmo_py = molmo_to_pixel(detection, res)
        nearest = min(visible_gt, key=lambda gt: math.hypot(molmo_px - gt.x, molmo_py - gt.y))
        add_error_row(
            rows,
            class_title=class_title,
            mesh_id=mesh_id,
            prompt=prompt,
            semantic_ids=semantic_ids,
            view_idx=view_idx,
            detection=detection,
            gt=nearest,
            match_mode="group_oracle",
            res=res,
            visibility_threshold_ratio=visibility_threshold_ratio,
        )
        counts["group_oracle_evaluable"] += 1

    # strict_single: only unambiguous single-semantic prompts are evaluated.
    if len(semantic_ids) == 1:
        sid = semantic_ids[0]
        strict_gt = [gt for gt in visible_gt if gt.semantic_id == sid]
        if strict_gt:
            gt = strict_gt[0]
            for detection in detections:
                add_error_row(
                    rows,
                    class_title=class_title,
                    mesh_id=mesh_id,
                    prompt=prompt,
                    semantic_ids=semantic_ids,
                    view_idx=view_idx,
                    detection=detection,
                    gt=gt,
                    match_mode="strict_single",
                    res=res,
                    visibility_threshold_ratio=visibility_threshold_ratio,
                )
                counts["strict_single_evaluable"] += 1
        else:
            counts["strict_single_occluded"] += len(detections)
    else:
        counts["strict_single_skipped_multisemantic"] += len(detections)

    return counts


def aggregate_mesh_view_rows(error_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in error_rows:
        key = (row["match_mode"], row["class_title"], row["mesh_id"], int(row["view_idx"]))
        grouped[key].append(float(row["error_px"]))

    out: list[dict[str, Any]] = []
    for (match_mode, class_title, mesh_id, view_idx), errors in sorted(grouped.items()):
        stats = finite_stats(errors)
        out.append({
            "match_mode": match_mode,
            "class_title": class_title,
            "mesh_id": mesh_id,
            "view_idx": view_idx,
            "num_visible_detections": stats["count"],
            "mean_error_px": stats["mean"],
            "median_error_px": stats["median"],
            "variance_error_px": stats["variance"],
            "std_error_px": stats["std"],
            "min_error_px": stats["min"],
            "max_error_px": stats["max"],
            "p25_error_px": stats["p25"],
            "p75_error_px": stats["p75"],
        })
    return out


def select_mesh_best_views(mesh_view_rows: list[dict[str, Any]], min_visible_detections: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mesh_view_rows:
        if int(row["num_visible_detections"] or 0) < min_visible_detections:
            continue
        grouped[(row["match_mode"], row["class_title"], row["mesh_id"])].append(row)

    best_rows: list[dict[str, Any]] = []
    for (match_mode, class_title, mesh_id), rows in sorted(grouped.items()):
        best = min(rows, key=lambda row: (float(row["mean_error_px"]), float(row["median_error_px"]), -int(row["num_visible_detections"])))
        best_rows.append({
            "match_mode": match_mode,
            "class_title": class_title,
            "mesh_id": mesh_id,
            "best_view_idx": int(best["view_idx"]),
            "best_view_mean_error_px": best["mean_error_px"],
            "best_view_median_error_px": best["median_error_px"],
            "best_view_variance_error_px": best["variance_error_px"],
            "best_view_min_error_px": best["min_error_px"],
            "best_view_max_error_px": best["max_error_px"],
            "best_view_num_visible_detections": best["num_visible_detections"],
        })
    return best_rows


def aggregate_class_best_stats(mesh_best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in mesh_best_rows:
        grouped[(row["match_mode"], row["class_title"])].append(float(row["best_view_mean_error_px"]))

    out: list[dict[str, Any]] = []
    for (match_mode, class_title), errors in sorted(grouped.items()):
        stats = finite_stats(errors)
        out.append({
            "match_mode": match_mode,
            "class_title": class_title,
            "num_meshes": stats["count"],
            "mean_best_view_error_px": stats["mean"],
            "median_best_view_error_px": stats["median"],
            "variance_best_view_error_px": stats["variance"],
            "std_best_view_error_px": stats["std"],
            "min_best_view_error_px": stats["min"],
            "max_best_view_error_px": stats["max"],
            "p25_best_view_error_px": stats["p25"],
            "p75_best_view_error_px": stats["p75"],
        })
    return out


def aggregate_top3_views_by_error(mesh_view_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in mesh_view_rows:
        grouped[(row["match_mode"], row["class_title"], int(row["view_idx"]))].append(row)

    ranked_by_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (match_mode, class_title, view_idx), rows in grouped.items():
        means = [float(row["mean_error_px"]) for row in rows]
        det_count = sum(int(row["num_visible_detections"] or 0) for row in rows)
        stats = finite_stats(means)
        ranked_by_class[(match_mode, class_title)].append({
            "match_mode": match_mode,
            "class_title": class_title,
            "view_idx": view_idx,
            "num_mesh_views": stats["count"],
            "num_visible_detections": det_count,
            "mean_mesh_view_error_px": stats["mean"],
            "median_mesh_view_error_px": stats["median"],
            "variance_mesh_view_error_px": stats["variance"],
            "min_mesh_view_error_px": stats["min"],
            "max_mesh_view_error_px": stats["max"],
        })

    out: list[dict[str, Any]] = []
    for key, rows in ranked_by_class.items():
        rows.sort(key=lambda row: (float(row["mean_mesh_view_error_px"]), -int(row["num_mesh_views"])))
        for rank, row in enumerate(rows[:3], start=1):
            out.append({"rank": rank, **row})
    return out


def aggregate_top3_best_view_frequency(mesh_best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in mesh_best_rows:
        grouped[(row["match_mode"], row["class_title"], int(row["best_view_idx"]))].append(float(row["best_view_mean_error_px"]))

    ranked_by_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (match_mode, class_title, view_idx), errors in grouped.items():
        stats = finite_stats(errors)
        ranked_by_class[(match_mode, class_title)].append({
            "match_mode": match_mode,
            "class_title": class_title,
            "view_idx": view_idx,
            "selected_mesh_count": stats["count"],
            "mean_selected_best_error_px": stats["mean"],
            "median_selected_best_error_px": stats["median"],
            "min_selected_best_error_px": stats["min"],
            "max_selected_best_error_px": stats["max"],
        })

    out: list[dict[str, Any]] = []
    for _key, rows in ranked_by_class.items():
        rows.sort(key=lambda row: (-int(row["selected_mesh_count"]), float(row["mean_selected_best_error_px"])))
        for rank, row in enumerate(rows[:3], start=1):
            out.append({"rank": rank, **row})
    return out


def aggregate_occlusion_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_title"]].append(row)

    out: list[dict[str, Any]] = []
    for class_title, cls_rows in sorted(grouped.items()):
        total = sum(int(row.get("total_molmo_predictions", 0)) for row in cls_rows)
        visible_prompts = sum(int(row.get("views_with_visible_target_and_prediction", 0)) for row in cls_rows)
        occluded = sum(int(row.get("occluded_gt_but_molmo_predicted", 0)) for row in cls_rows)
        no_gt = sum(int(row.get("no_gt_annotation_predictions", 0)) for row in cls_rows)
        out.append({
            "class_title": class_title,
            "num_meshes_seen": len({row["mesh_id"] for row in cls_rows}),
            "total_molmo_predictions": total,
            "views_with_visible_target_and_prediction": visible_prompts,
            "occluded_gt_but_molmo_predicted": occluded,
            "no_gt_annotation_predictions": no_gt,
            "occluded_prediction_rate": (occluded / total) if total else None,
        })
    return out


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KEYPOINT_DATASET_PATH"] = str(args.keypointnet_dir)

    # Import after KEYPOINT_DATASET_PATH is set because kp_utils.data.utils resolves
    # dataset paths at import time.
    from kp_utils.data.keypoint_labels import INVERSE_CLASS_MAPPING  # noqa: PLC0415
    from kp_utils.data.utils import load_keypoints  # noqa: PLC0415

    global torch, IO, Meshes, trimesh, sample_view_points, camera_from_eye_at_up, setup_renderer
    import torch as torch_module  # noqa: PLC0415
    from pytorch3d.io import IO as IOClass  # noqa: PLC0415
    from pytorch3d.structures import Meshes as MeshesClass  # noqa: PLC0415
    import trimesh as trimesh_module  # noqa: PLC0415
    from kp_utils.rendering import sample_view_points as sample_view_points_func  # noqa: PLC0415
    from kp_utils.rendering import setup_renderer as setup_renderer_func  # noqa: PLC0415
    from zerokey.rendering import camera_from_eye_at_up as camera_from_eye_at_up_func  # noqa: PLC0415

    torch = torch_module
    IO = IOClass
    Meshes = MeshesClass
    trimesh = trimesh_module
    sample_view_points = sample_view_points_func
    camera_from_eye_at_up = camera_from_eye_at_up_func
    setup_renderer = setup_renderer_func

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    io = IO()
    all_keypoints = load_keypoints()
    views = sample_view_points(args.view_radius, args.view_partition)
    renderer = setup_renderer(device, res=args.res) if args.ray_engine == "zbuf" else None

    mesh_cache: dict[Path, Meshes] = {}
    trimesh_cache: dict[Path, trimesh.Trimesh] = {}
    ray_intersector_cache: dict[Path, Any] = {}
    zbuf_cache: dict[Path, Any] = {}
    camera_cache: dict[Path, CamerasBase] = {}

    error_rows: list[dict[str, Any]] = []
    prompt_view_rows: list[dict[str, Any]] = []

    print("=" * 120)
    print("[AUDIT] Molmo2D vs visible GT best-view statistics")
    print(f"result_dir={args.result_dir}")
    print(f"keypointnet_dir={args.keypointnet_dir}")
    print(f"out_dir={args.out_dir}")
    print(f"classes={','.join(args.classes)}")
    print(f"res={args.res}")
    print(f"visibility_threshold_ratio={args.visibility_threshold_ratio}")
    print(f"ray_engine={args.ray_engine}")
    if args.ray_engine == "zbuf":
        print(f"zbuf_depth_threshold={args.zbuf_depth_threshold}")
        print(f"zbuf_window_radius={args.zbuf_window_radius}")
        print(f"render_batch_size={args.render_batch_size}")
    print(f"min_visible_detections_per_view={args.min_visible_detections_per_view}")
    print("=" * 120)

    for class_title in args.classes:
        synset = INVERSE_CLASS_MAPPING.get(class_title)
        if synset is None:
            print(f"[WARN] unknown class, skipping: {class_title}")
            continue
        class_dir = args.result_dir / class_title
        if not class_dir.exists():
            print(f"[WARN] result class dir missing, skipping: {class_dir}")
            continue
        mesh_dirs = [p for p in sorted(class_dir.iterdir()) if p.is_dir()]
        if args.max_meshes_per_class > 0:
            mesh_dirs = mesh_dirs[:args.max_meshes_per_class]

        print("-" * 120)
        print(f"[CLASS] {class_title}: candidate mesh dirs={len(mesh_dirs)}")

        for mesh_idx, mesh_dir in enumerate(mesh_dirs, start=1):
            mesh_id = mesh_dir.name
            json_files = sorted(mesh_dir.glob("Molmo2D_*_kps2d.json"))
            if not json_files:
                continue
            mesh_path = mesh_path_for(args.keypointnet_dir, args.mesh_subdir, synset, mesh_id)
            if not mesh_path.exists():
                print(f"[WARN] mesh not found, skipping {class_title}/{mesh_id}: {mesh_path}")
                continue
            gt_kps = all_keypoints.get(synset, {}).get(mesh_id, [])
            if not gt_kps:
                print(f"[WARN] no GT keypoints, skipping {class_title}/{mesh_id}")
                continue

            if mesh_path not in mesh_cache:
                mesh_obj = io.load_mesh(mesh_path, device=device, include_textures=False)
                if not isinstance(mesh_obj, Meshes):
                    mesh_obj = Meshes(verts=[mesh_obj.verts_packed()], faces=[mesh_obj.faces_packed()])
                mesh_cache[mesh_path] = mesh_obj.to(device)
                camera_cache[mesh_path] = get_camera_batch(mesh_cache[mesh_path], views, device)
                if args.ray_engine == "zbuf":
                    assert renderer is not None
                    zbuf_cache[mesh_path] = render_zbuf_views(renderer, mesh_cache[mesh_path], camera_cache[mesh_path], args.render_batch_size)
                    trimesh_cache[mesh_path] = None
                    ray_intersector_cache[mesh_path] = None
                else:
                    mesh_tri = trimesh.load(str(mesh_path), force="mesh")
                    trimesh_cache[mesh_path] = mesh_tri
                    if args.ray_engine == "triangle":
                        from trimesh.ray.ray_triangle import RayMeshIntersector  # noqa: PLC0415
                        ray_intersector_cache[mesh_path] = RayMeshIntersector(mesh_tri)
                    else:
                        # pyembree can be faster, but has caused native segfaults in some environments.
                        ray_intersector_cache[mesh_path] = mesh_tri.ray

            mesh = mesh_cache[mesh_path]
            mesh_tri = trimesh_cache[mesh_path]
            ray_intersector = ray_intersector_cache[mesh_path]
            zbuf_views = zbuf_cache.get(mesh_path)
            cameras = camera_cache[mesh_path]
            camera_centers = cameras.get_camera_center().detach().cpu().numpy()

            if mesh_idx % 50 == 0 or mesh_idx == 1:
                print(f"  [{mesh_idx:4d}/{len(mesh_dirs):4d}] {mesh_id} jsons={len(json_files)}")

            for json_path in json_files:
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[WARN] failed to read {json_path}: {exc}")
                    continue
                semantic_ids = semantic_ids_from_payload(payload)
                prompt = str(payload.get("prompt", ""))
                kps_2d = payload.get("kps_2d", {})
                if not isinstance(kps_2d, dict):
                    continue

                prompt_total_predictions = 0
                prompt_occluded_predictions = 0
                prompt_no_gt_predictions = 0
                prompt_visible_views_with_prediction = 0

                gt_points = collect_gt_points(gt_kps, set(semantic_ids))

                for view_idx_str, view_payload in kps_2d.items():
                    try:
                        view_idx = int(view_idx_str)
                    except (TypeError, ValueError):
                        continue
                    if view_idx < 0 or view_idx >= len(views):
                        continue
                    detections = flatten_molmo_detections(view_payload)
                    if not detections:
                        continue
                    prompt_total_predictions += len(detections)

                    if not gt_points:
                        prompt_no_gt_predictions += len(detections)
                        continue

                    camera = cameras[view_idx]
                    visible_gt = project_visible_gt(
                        gt_points,
                        camera,
                        camera_centers[view_idx],
                        ray_intersector,
                        None if mesh_tri is None else mesh_tri.bounds,
                        None if zbuf_views is None else zbuf_views[view_idx],
                        args.ray_engine,
                        args.res,
                        args.visibility_threshold_ratio,
                        args.zbuf_depth_threshold,
                        args.zbuf_window_radius,
                        device,
                    )
                    if visible_gt:
                        prompt_visible_views_with_prediction += 1

                    counts = evaluate_detection_matches(
                        error_rows,
                        class_title=class_title,
                        mesh_id=mesh_id,
                        prompt=prompt,
                        semantic_ids=semantic_ids,
                        view_idx=view_idx,
                        detections=detections,
                        visible_gt=visible_gt,
                        res=args.res,
                        visibility_threshold_ratio=args.visibility_threshold_ratio,
                    )
                    prompt_occluded_predictions += int(counts.get("occluded_gt_but_molmo_predicted", 0))

                prompt_view_rows.append({
                    "class_title": class_title,
                    "mesh_id": mesh_id,
                    "json_file": json_path.name,
                    "prompt": prompt,
                    "semantic_ids": ",".join(map(str, semantic_ids)),
                    "num_semantic_ids": len(semantic_ids),
                    "total_molmo_predictions": prompt_total_predictions,
                    "views_with_visible_target_and_prediction": prompt_visible_views_with_prediction,
                    "occluded_gt_but_molmo_predicted": prompt_occluded_predictions,
                    "no_gt_annotation_predictions": prompt_no_gt_predictions,
                })

    mesh_view_rows = aggregate_mesh_view_rows(error_rows)
    mesh_best_rows = select_mesh_best_views(mesh_view_rows, args.min_visible_detections_per_view)
    class_best_rows = aggregate_class_best_stats(mesh_best_rows)
    class_top3_error_rows = aggregate_top3_views_by_error(mesh_view_rows)
    class_top3_freq_rows = aggregate_top3_best_view_frequency(mesh_best_rows)
    occlusion_rows = aggregate_occlusion_stats(prompt_view_rows)

    write_csv(
        args.out_dir / "molmo2d_vs_visible_gt_rows.csv",
        error_rows,
        [
            "class_title", "mesh_id", "prompt", "semantic_ids", "view_idx", "detection_index",
            "match_mode", "matched_semantic_id", "molmo_px", "molmo_py", "gt_px", "gt_py", "gt_pz",
            "error_px", "visibility_threshold_ratio",
        ],
    )
    write_csv(
        args.out_dir / "prompt_visibility_occlusion_stats.csv",
        prompt_view_rows,
        [
            "class_title", "mesh_id", "json_file", "prompt", "semantic_ids", "num_semantic_ids",
            "total_molmo_predictions", "views_with_visible_target_and_prediction",
            "occluded_gt_but_molmo_predicted", "no_gt_annotation_predictions",
        ],
    )
    write_csv(
        args.out_dir / "mesh_view_error_summary.csv",
        mesh_view_rows,
        [
            "match_mode", "class_title", "mesh_id", "view_idx", "num_visible_detections",
            "mean_error_px", "median_error_px", "variance_error_px", "std_error_px",
            "min_error_px", "max_error_px", "p25_error_px", "p75_error_px",
        ],
    )
    write_csv(
        args.out_dir / "mesh_best_view_summary.csv",
        mesh_best_rows,
        [
            "match_mode", "class_title", "mesh_id", "best_view_idx", "best_view_num_visible_detections",
            "best_view_mean_error_px", "best_view_median_error_px", "best_view_variance_error_px",
            "best_view_min_error_px", "best_view_max_error_px",
        ],
    )
    write_csv(
        args.out_dir / "class_best_view_stats.csv",
        class_best_rows,
        [
            "match_mode", "class_title", "num_meshes", "mean_best_view_error_px",
            "median_best_view_error_px", "variance_best_view_error_px", "std_best_view_error_px",
            "min_best_view_error_px", "max_best_view_error_px", "p25_best_view_error_px", "p75_best_view_error_px",
        ],
    )
    write_csv(
        args.out_dir / "class_top3_views_by_error.csv",
        class_top3_error_rows,
        [
            "rank", "match_mode", "class_title", "view_idx", "num_mesh_views", "num_visible_detections",
            "mean_mesh_view_error_px", "median_mesh_view_error_px", "variance_mesh_view_error_px",
            "min_mesh_view_error_px", "max_mesh_view_error_px",
        ],
    )
    write_csv(
        args.out_dir / "class_top3_best_view_frequency.csv",
        class_top3_freq_rows,
        [
            "rank", "match_mode", "class_title", "view_idx", "selected_mesh_count",
            "mean_selected_best_error_px", "median_selected_best_error_px",
            "min_selected_best_error_px", "max_selected_best_error_px",
        ],
    )
    write_csv(
        args.out_dir / "class_occlusion_prediction_stats.csv",
        occlusion_rows,
        [
            "class_title", "num_meshes_seen", "total_molmo_predictions", "views_with_visible_target_and_prediction",
            "occluded_gt_but_molmo_predicted", "no_gt_annotation_predictions", "occluded_prediction_rate",
        ],
    )

    summary = {
        "result_dir": str(args.result_dir),
        "keypointnet_dir": str(args.keypointnet_dir),
        "classes": list(args.classes),
        "res": args.res,
        "view_radius": args.view_radius,
        "view_partition": args.view_partition,
        "visibility_threshold_ratio": args.visibility_threshold_ratio,
        "ray_engine": args.ray_engine,
        "zbuf_depth_threshold": args.zbuf_depth_threshold,
        "zbuf_window_radius": args.zbuf_window_radius,
        "render_batch_size": args.render_batch_size,
        "min_visible_detections_per_view": args.min_visible_detections_per_view,
        "num_error_rows": len(error_rows),
        "num_mesh_view_rows": len(mesh_view_rows),
        "num_mesh_best_rows": len(mesh_best_rows),
        "class_best_view_stats": class_best_rows,
        "class_occlusion_prediction_stats": occlusion_rows,
        "outputs": {
            "molmo2d_vs_visible_gt_rows": str(args.out_dir / "molmo2d_vs_visible_gt_rows.csv"),
            "prompt_visibility_occlusion_stats": str(args.out_dir / "prompt_visibility_occlusion_stats.csv"),
            "mesh_view_error_summary": str(args.out_dir / "mesh_view_error_summary.csv"),
            "mesh_best_view_summary": str(args.out_dir / "mesh_best_view_summary.csv"),
            "class_best_view_stats": str(args.out_dir / "class_best_view_stats.csv"),
            "class_top3_views_by_error": str(args.out_dir / "class_top3_views_by_error.csv"),
            "class_top3_best_view_frequency": str(args.out_dir / "class_top3_best_view_frequency.csv"),
            "class_occlusion_prediction_stats": str(args.out_dir / "class_occlusion_prediction_stats.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 120)
    print("[DONE]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 120)
    return summary


def main() -> None:
    args = parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
