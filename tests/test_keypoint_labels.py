"""Tests for kp_utils.data.keypoint_labels — class mappings and label metadata."""
from __future__ import annotations

import re

from kp_utils.data.keypoint_labels import (
    CLASS_MAPPING,
    INVERSE_CLASS_MAPPING,
    KEYPOINT_LABELS,
)


def test_class_mapping_has_16_entries() -> None:
    assert len(CLASS_MAPPING) == 16


def test_class_mapping_keys_are_8digit_strings() -> None:
    pattern = re.compile(r"^\d{8}$")
    for key in CLASS_MAPPING:
        assert pattern.match(key), f"{key!r} does not match 8-digit pattern"


def test_class_mapping_values_nonempty_strings() -> None:
    for value in CLASS_MAPPING.values():
        assert isinstance(value, str)
        assert len(value) > 0


def test_inverse_class_mapping_roundtrip() -> None:
    for synset_id, class_name in CLASS_MAPPING.items():
        assert INVERSE_CLASS_MAPPING[class_name] == synset_id


def test_inverse_class_mapping_contains_all_class_mapping() -> None:
    class_names = set(CLASS_MAPPING.values())
    assert class_names.issubset(INVERSE_CLASS_MAPPING.keys())


def test_keypoint_labels_categories_subset() -> None:
    valid_names = set(CLASS_MAPPING.values())
    for category in KEYPOINT_LABELS:
        assert category in valid_names


def test_keypoint_labels_entries_are_tuples() -> None:
    for category, entries in KEYPOINT_LABELS.items():
        for idx, entry in entries.items():
            assert isinstance(entry, tuple), f"{category}[{idx}]"
            assert len(entry) == 3, f"{category}[{idx}]"


def test_keypoint_labels_tuple_types() -> None:
    for category, entries in KEYPOINT_LABELS.items():
        for idx, (desc, flag1, flag2) in entries.items():
            assert isinstance(desc, str), f"{category}[{idx}][0]"
            assert isinstance(flag1, bool), f"{category}[{idx}][1]"
            assert isinstance(flag2, bool), f"{category}[{idx}][2]"
