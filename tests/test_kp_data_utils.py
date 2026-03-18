"""Tests for kp_utils.data.utils — naive PCD reader."""
from __future__ import annotations

import os

import numpy as np
import pytest

from kp_utils.data.utils import naive_read_pcd

_PCD_HEADER = """\
# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH {width}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {width}
DATA ascii
"""

_RED = 16711680
_GREEN = 65280
_BLUE = 255

_SAMPLE_ROWS = [
    "1.0 2.0 3.0 {red}",
    "4.0 5.0 6.0 {green}",
    "7.0 8.0 9.0 {blue}",
]


def _write_pcd(directory: str, rows: list[str] | None = None) -> str:
    """Write a temporary PCD file and return its path."""
    if rows is None:
        rows = [r.format(red=_RED, green=_GREEN, blue=_BLUE) for r in _SAMPLE_ROWS]
    header = _PCD_HEADER.format(width=len(rows))
    path = os.path.join(directory, "test.pcd")
    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(rows))
        f.write("\n")
    return path


def test_naive_read_pcd_shape(tmp_path: pytest.TempPathFactory) -> None:
    path = _write_pcd(str(tmp_path))
    pc, _colors = naive_read_pcd(path)
    assert pc.shape == (3, 3)


def test_naive_read_pcd_dtype(tmp_path: pytest.TempPathFactory) -> None:
    path = _write_pcd(str(tmp_path))
    pc, _colors = naive_read_pcd(path)
    assert np.issubdtype(pc.dtype, np.floating)


def test_naive_read_pcd_colors_shape(tmp_path: pytest.TempPathFactory) -> None:
    path = _write_pcd(str(tmp_path))
    _pc, colors = naive_read_pcd(path)
    assert colors.shape == (3, 3)


def test_naive_read_pcd_color_extraction(tmp_path: pytest.TempPathFactory) -> None:
    path = _write_pcd(str(tmp_path))
    _pc, colors = naive_read_pcd(path)
    np.testing.assert_array_equal(colors[0], [255, 0, 0])
    np.testing.assert_array_equal(colors[1], [0, 255, 0])
    np.testing.assert_array_equal(colors[2], [0, 0, 255])


def test_naive_read_pcd_empty_data(tmp_path: pytest.TempPathFactory) -> None:
    """A PCD file with a header but no data points raises ValueError."""
    path = _write_pcd(str(tmp_path), rows=[])
    with pytest.raises(ValueError):
        naive_read_pcd(path)
