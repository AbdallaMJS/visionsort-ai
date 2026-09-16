from pathlib import Path

import pytest

from visionsort.data import ImageRecord, LABELS, stratified_split


def records_per_class(count: int = 10):
    rows = []
    for label in LABELS:
        for i in range(count):
            rows.append(ImageRecord(Path(f"/{label}/{i}.jpg"), label))
    return rows


def test_stratified_split_preserves_every_class():
    split = stratified_split(records_per_class(), seed=7)
    for name in ("train", "validation", "test"):
        labels = {record.label for record in split[name]}
        assert labels == set(LABELS)


def test_split_is_reproducible():
    first = stratified_split(records_per_class(), seed=42)
    second = stratified_split(records_per_class(), seed=42)
    assert first == second


def test_split_sizes_cover_all_records():
    records = records_per_class()
    split = stratified_split(records)
    assert sum(len(values) for values in split.values()) == len(records)


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        stratified_split([ImageRecord(Path("/x.jpg"), "unknown")])
