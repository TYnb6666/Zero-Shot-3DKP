"""Tests for kp_utils.geometry — face adjacency and geodesic distances."""
from __future__ import annotations

import pytest
import torch

from kp_utils.geometry import find_adjacent_faces

try:
    from kp_utils.geometry import (
        pairwise_geodesic_distances,
        pairwise_geodesic_distances_mesh,
    )
    _HAS_PP3D = True
except ImportError:
    _HAS_PP3D = False


def test_find_adjacent_single_triangle() -> None:
    faces = torch.tensor([[0, 1, 2]])
    adj = find_adjacent_faces(faces)
    if adj.numel() == 0:
        assert adj.shape[0] == 1
    else:
        assert torch.all(adj == -1)


def test_find_adjacent_shared_edge() -> None:
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]])
    adj = find_adjacent_faces(faces)
    assert 1 in adj[0].tolist()
    assert 0 in adj[1].tolist()


def test_find_adjacent_tetrahedron() -> None:
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    adj = find_adjacent_faces(faces)
    for i in range(4):
        neighbors = adj[i][adj[i] >= 0].tolist()
        assert sorted(neighbors) == sorted(j for j in range(4) if j != i)


def test_find_adjacent_output_type() -> None:
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]])
    adj = find_adjacent_faces(faces)
    assert isinstance(adj, torch.Tensor)


def test_find_adjacent_padding() -> None:
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3]])
    adj = find_adjacent_faces(faces)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            val = adj[i, j].item()
            assert val == -1 or 0 <= val < faces.shape[0]


def test_find_adjacent_shape() -> None:
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    adj = find_adjacent_faces(faces)
    m = faces.shape[0]
    assert adj.shape[0] == m
    assert adj.shape[1] >= 1


@pytest.mark.skipif(not _HAS_PP3D, reason="requires potpourri3d")
def test_pairwise_geodesic_distances_shape() -> None:
    n = 200
    points = torch.randn(n, 3)
    dists = pairwise_geodesic_distances(points)
    assert dists.shape == (n, n)


@pytest.mark.skipif(not _HAS_PP3D, reason="requires potpourri3d")
def test_pairwise_geodesic_distances_mesh_shape() -> None:
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    dists = pairwise_geodesic_distances_mesh(verts, faces)
    n = verts.shape[0]
    assert dists.shape == (n, n)
