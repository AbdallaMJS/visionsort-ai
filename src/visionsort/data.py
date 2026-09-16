from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

LABELS = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def download_trashnet(destination: str | Path = "data") -> Path:
    """Download the resized TrashNet archive from its Hugging Face dataset repo."""
    from huggingface_hub import hf_hub_download

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id="garythung/trashnet",
            filename="dataset-resized.zip",
            repo_type="dataset",
        )
    )
    extracted = destination / "trashnet"
    if not extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    return find_dataset_root(extracted)


def find_dataset_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = [root, root / "dataset-resized"]
    candidates.extend(path for path in root.rglob("dataset-resized") if path.is_dir())
    for candidate in candidates:
        if all((candidate / label).is_dir() for label in LABELS):
            return candidate
    raise FileNotFoundError("Could not locate TrashNet class folders.")


def discover_records(root: str | Path) -> list[ImageRecord]:
    root = find_dataset_root(root)
    records = []
    for label in LABELS:
        for path in sorted((root / label).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(ImageRecord(path=path, label=label))
    if not records:
        raise ValueError("No images found in TrashNet dataset root.")
    return records


def stratified_split(
    records: list[ImageRecord],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Split each class independently so minority classes remain represented."""
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must leave a test set.")
    groups: dict[str, list[ImageRecord]] = {label: [] for label in LABELS}
    for record in records:
        if record.label not in groups:
            raise ValueError(f"Unknown label: {record.label}")
        groups[record.label].append(record)

    rng = random.Random(seed)
    split = {"train": [], "validation": [], "test": []}
    for label, items in groups.items():
        if not items:
            continue
        items = items.copy()
        rng.shuffle(items)
        train_end = int(len(items) * train_fraction)
        val_end = int(len(items) * (train_fraction + validation_fraction))
        split["train"].extend(items[:train_end])
        split["validation"].extend(items[train_end:val_end])
        split["test"].extend(items[val_end:])
    return split
