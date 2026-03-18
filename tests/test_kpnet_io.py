"""Tests for zerokey.io.kpnet.setids_to_uint8_flags."""
from __future__ import annotations

import pytest
import torch

from tests import assert_tensor_close


def _call(setids: list[int]) -> torch.Tensor:
    from zerokey.io.kpnet import setids_to_uint8_flags

    return setids_to_uint8_flags(setids)


def test_empty_setids() -> None:
    result = _call([])
    assert_tensor_close(result, torch.tensor([0, 0, 0], dtype=torch.uint8))


def test_single_id_0() -> None:
    result = _call([0])
    assert int(result[-1].item()) & 1 == 1


def test_single_id_23() -> None:
    result = _call([23])
    assert int(result[0].item()) & 0x80 == 0x80


def test_multiple_ids() -> None:
    result = _call([0, 3, 7])
    expected_flags = (1 << 0) | (1 << 3) | (1 << 7)
    actual_int = int.from_bytes(result.tolist(), byteorder="big")
    assert actual_int == expected_flags


def test_dtype_uint8() -> None:
    result = _call([0])
    assert result.dtype == torch.uint8


def test_shape_3() -> None:
    result = _call([5])
    assert result.shape == (3,)


def test_roundtrip() -> None:
    ids = [2, 5, 10, 20]
    result = _call(ids)
    recovered = int.from_bytes(result.tolist(), byteorder="big")
    for i in ids:
        assert recovered & (1 << i), f"bit {i} not set"


def test_negative_id_raises() -> None:
    with pytest.raises(ValueError):
        _call([-1])


def test_id_24_raises() -> None:
    with pytest.raises(ValueError):
        _call([24])


def test_duplicate_ids_ok() -> None:
    single = _call([1])
    double = _call([1, 1])
    assert_tensor_close(single, double)


def test_all_ids() -> None:
    result = _call(list(range(24)))
    actual_int = int.from_bytes(result.tolist(), byteorder="big")
    assert actual_int == 0xFFFFFF


def test_float_id_raises() -> None:
    with pytest.raises(ValueError):
        _call([1.0])  # type: ignore[list-item]
