"""Tests for zerokey.vis.base.VisGeneratorBase.aggregate_kps."""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from unittest import mock


def _make_vis_gen():
    """Create a VisGeneratorBase instance without calling __init__."""
    from zerokey.vis.base import VisGeneratorBase

    obj = VisGeneratorBase.__new__(VisGeneratorBase)
    obj.vis = False
    obj.io = mock.MagicMock()
    return obj


def test_aggregate_kps_single_cluster() -> None:
    gen = _make_vis_gen()
    pts = torch.randn(20, 3) * 0.01 + torch.tensor([1.0, 2.0, 3.0])
    pts_3d = Pointclouds(pts[None])
    result = gen.aggregate_kps(mesh=None, pts_3d=pts_3d)
    result_pts = result.points_packed()
    assert result_pts is not None
    assert result_pts.shape[0] >= 1
    mean_pt = pts.mean(dim=0)
    centroid = result_pts[0]
    np.testing.assert_allclose(centroid.cpu().numpy(), mean_pt.numpy(), atol=0.1)


def test_aggregate_kps_two_clusters() -> None:
    gen = _make_vis_gen()
    group_a = torch.randn(15, 3) * 0.01 + torch.tensor([0.0, 0.0, 0.0])
    group_b = torch.randn(15, 3) * 0.01 + torch.tensor([10.0, 10.0, 10.0])
    pts = torch.cat([group_a, group_b], dim=0)
    pts_3d = Pointclouds(pts[None])
    result = gen.aggregate_kps(mesh=None, pts_3d=pts_3d)
    result_pts = result.points_packed()
    assert result_pts is not None
    assert result_pts.shape[0] >= 2


def test_aggregate_kps_returns_pointclouds() -> None:
    gen = _make_vis_gen()
    pts = torch.randn(10, 3)
    pts_3d = Pointclouds(pts[None])
    result = gen.aggregate_kps(mesh=None, pts_3d=pts_3d)
    assert isinstance(result, Pointclouds)
