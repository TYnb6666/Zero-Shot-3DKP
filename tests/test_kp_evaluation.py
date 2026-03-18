"""Tests for kp_utils.evaluation — geodesic distances and IoU evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from kp_utils.evaluation import eval_det_cls, eval_iou, gen_geo_dists


@pytest.fixture(scope="module")
def grid_pc_and_dists():
    xs = np.linspace(0, 1, 6)
    ys = np.linspace(0, 1, 5)
    grid = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    dists = gen_geo_dists(grid)
    return grid, dists


def test_gen_geo_dists_shape(grid_pc_and_dists) -> None:
    pc, dists = grid_pc_and_dists
    n = pc.shape[0]
    assert dists.shape == (n, n)


def test_gen_geo_dists_diagonal_near_zero(grid_pc_and_dists) -> None:
    _pc, dists = grid_pc_and_dists
    diag = np.diag(dists)
    np.testing.assert_allclose(diag, 0.0, atol=1e-12)


def test_gen_geo_dists_symmetric(grid_pc_and_dists) -> None:
    _pc, dists = grid_pc_and_dists
    np.testing.assert_allclose(dists, dists.T, atol=1e-12)


def test_gen_geo_dists_nonneg(grid_pc_and_dists) -> None:
    _pc, dists = grid_pc_and_dists
    finite = dists.copy()
    finite[np.isinf(finite)] = 0.0
    assert np.all(finite >= 0.0)


def _make_geo_dists(n: int) -> np.ndarray:
    """Create a simple Euclidean-distance matrix for *n* collinear points."""
    pts = np.linspace(0, 1, n).reshape(-1, 1)
    return np.abs(pts - pts.T)


def test_eval_det_cls_perfect_match() -> None:
    n = 50
    geo = _make_geo_dists(n)
    gt_indices = [0, 10, 20]
    pred = {"mesh0": {"indices": gt_indices, "confidence": [1.0] * 3}}
    gt = {"mesh0": gt_indices}
    geo_dists = {"mesh0": geo}
    score = eval_det_cls(pred, gt, geo_dists, dist_thresh=0.5)
    assert abs(score - 1.0) < 1e-5


def test_eval_det_cls_no_prediction() -> None:
    n = 50
    geo = _make_geo_dists(n)
    pred = {"mesh0": {"indices": [], "confidence": []}}
    gt = {"mesh0": [0, 10, 20]}
    geo_dists = {"mesh0": geo}
    score = eval_det_cls(pred, gt, geo_dists, dist_thresh=0.5)
    assert abs(score - 0.0) < 1e-5


def test_eval_det_cls_partial_match() -> None:
    n = 50
    geo = _make_geo_dists(n)
    gt_indices = [0, 10, 20]
    pred_indices = [0, 10, 49]
    pred = {"mesh0": {"indices": pred_indices, "confidence": [1.0] * 3}}
    gt = {"mesh0": gt_indices}
    geo_dists = {"mesh0": geo}
    score = eval_det_cls(pred, gt, geo_dists, dist_thresh=0.05)
    assert score > 0.0
    assert score < 1.0


def test_eval_det_cls_confidence_threshold() -> None:
    n = 50
    geo = _make_geo_dists(n)
    gt_indices = [0, 10, 20]
    pred = {"mesh0": {"indices": [0, 10, 49], "confidence": [0.9, 0.8, 0.1]}}
    gt = {"mesh0": gt_indices}
    geo_dists = {"mesh0": geo}
    score_hi = eval_det_cls(pred, gt, geo_dists, dist_thresh=0.05, confidence_thresh=0.5)
    score_lo = eval_det_cls(pred, gt, geo_dists, dist_thresh=0.05, confidence_thresh=0.0)
    assert score_hi >= score_lo


def test_eval_iou_returns_dict() -> None:
    n = 50
    geo = _make_geo_dists(n)
    gt_all = {"chair": {"mesh0": [0, 10, 20]}}
    pred_all = {"chair": {"mesh0": {"indices": [0, 10, 20], "confidence": [1.0] * 3}}}
    geo_dists = {"mesh0": geo}
    result = eval_iou(pred_all, gt_all, geo_dists, dist_thresh=0.5)
    assert isinstance(result, dict)
    assert "chair" in result


def test_eval_iou_multiple_classes() -> None:
    n = 50
    geo = _make_geo_dists(n)
    geo_dists = {"mesh0": geo}
    gt_all = {
        "chair": {"mesh0": [0, 10, 20]},
        "table": {"mesh0": [5, 15, 25]},
    }
    pred_all = {
        "chair": {"mesh0": {"indices": [0, 10, 20], "confidence": [1.0] * 3}},
        "table": {"mesh0": {"indices": [5, 15, 25], "confidence": [1.0] * 3}},
    }
    result = eval_iou(pred_all, gt_all, geo_dists, dist_thresh=0.5)
    assert set(result.keys()) == {"chair", "table"}
    for cls_name, score in result.items():
        assert abs(score - 1.0) < 1e-5, cls_name
