"""Tests for kp_utils.rendering — viewpoint sampling and point rotation."""
from __future__ import annotations

import numpy as np

from kp_utils.rendering import rotate_points, sample_view_points


def _expected_count(partition: int) -> int:
    return (partition + 1) * 2 * partition + 2


def test_sample_view_points_shape() -> None:
    partition = 2
    pts = sample_view_points(radius=1.0, partition=partition)
    expected_n = _expected_count(partition)
    assert pts.shape == (expected_n, 3)


def test_sample_view_points_radius() -> None:
    radius = 5.0
    pts = sample_view_points(radius=radius, partition=3)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, radius, atol=0.01)


def test_sample_view_points_includes_poles() -> None:
    radius = 2.0
    pts = sample_view_points(radius=radius, partition=2)
    top_dists = np.linalg.norm(pts - np.array([0, radius, 0]), axis=1)
    bot_dists = np.linalg.norm(pts - np.array([0, -radius, 0]), axis=1)
    assert top_dists.min() < 0.01
    assert bot_dists.min() < 0.01


def test_sample_view_points_partition_scaling() -> None:
    pts2 = sample_view_points(radius=1.0, partition=2)
    pts3 = sample_view_points(radius=1.0, partition=3)
    assert len(pts3) > len(pts2)


def test_rotate_points_identity() -> None:
    points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    result = rotate_points(points, angle=0.0)
    for orig, rot in zip(points, result):
        np.testing.assert_allclose(rot, orig, atol=1e-10)


def test_rotate_points_90deg() -> None:
    result = rotate_points([[1.0, 0.0, 0.0]], angle=np.pi / 2, axis=[0, 0, 1])
    np.testing.assert_allclose(result[0], [0.0, 1.0, 0.0], atol=1e-10)


def test_rotate_points_magnitude_preserved() -> None:
    points = [[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    result = rotate_points(points, angle=1.23, axis=[1, 1, 0])
    for orig, rot in zip(points, result):
        np.testing.assert_allclose(np.linalg.norm(rot), np.linalg.norm(orig), atol=1e-10)


def test_rotate_points_custom_axis() -> None:
    result = rotate_points([[1.0, 0.0, 0.0]], angle=np.pi, axis=[0, 0, 1])
    np.testing.assert_allclose(result[0], [-1.0, 0.0, 0.0], atol=1e-10)
