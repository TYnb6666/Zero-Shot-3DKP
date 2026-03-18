"""Furthest Point Sampling (FPS) on CPU using numpy.

Provides a pure-numpy implementation of greedy FPS that iteratively selects
the point farthest from the already-chosen set until *k* points are picked.
Used by :pyfunc:`inference.extract_patch_features.resample_points` to
downsample raw point clouds before feeding them to the patch encoder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fps_indices(xyz: NDArray[np.floating], k: int, *, seed: int = 0) -> NDArray[np.int64]:
    """Select *k* points from *xyz* via Furthest Point Sampling.

    Starting from a random seed point, each subsequent point is chosen as
    the one whose minimum squared-distance to *all* previously selected
    points is largest.  Runs in O(N * k) time.

    Args:
        xyz: Point cloud of shape ``(N, 3)``.
        k: Number of points to sample.
        seed: Random seed used to pick the initial point.

    Returns:
        Integer index array of shape ``(k,)`` into *xyz*.
        If ``N <= k`` the full ``np.arange(N)`` is returned.
    """
    N: int = xyz.shape[0]
    if N <= k:
        return np.arange(N, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idxs = np.empty(k, dtype=np.int64)
    idxs[0] = int(rng.randint(0, N))
    dists: NDArray[np.float64] = np.full(N, np.inf, dtype=np.float64)
    last = xyz[idxs[0]][None, :]
    dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    for i in range(1, k):
        idxs[i] = int(np.argmax(dists))
        last = xyz[idxs[i]][None, :]
        dists = np.minimum(dists, np.sum((xyz - last) ** 2, axis=1))
    return idxs
