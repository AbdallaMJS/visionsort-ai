from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import LABELS, discover_records, stratified_split
from .model import WasteDataset, build_mobilenet, build_small_cnn


def _run_epoch(model, loader, loss_fn, optimizer, device: str, train: bool):
    import torch

    model.train(train)
    losses = []
    y_true: list[int] = []
    y_pred: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_model(
    data_root: str | Path,
    architecture: str = "mobilenet",
    epochs: int = 8,
    batch_size: int = 32,
    output_dir: str | Path = "artifacts",
) -> dict:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = discover_records(data_root)
    split = stratified_split(records, seed=42)

    loaders = {
        name: DataLoader(
            WasteDataset(items, train=name == "train"),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=2,
        )
        for name, items in split.items()
    }

    if architecture == "mobilenet":
        model = build_mobilenet(pretrained=True)
        learning_rate = 1e-4
    elif architecture == "cnn":
        model = build_small_cnn()
        learning_rate = 1e-3
    else:
        raise ValueError("architecture must be 'mobilenet' or 'cnn'")

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state = deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, loaders["train"], loss_fn, optimizer, device, train=True)
        val_metrics = _run_epoch(model, loaders["validation"], loss_fn, optimizer, device, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": val_metrics["loss"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test = _run_epoch(model, loaders["test"], loss_fn, optimizer, device, train=False)
    report = classification_report(
        test["y_true"],
        test["y_pred"],
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(len(LABELS))))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / f"{architecture}.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    summary = {
        "architecture": architecture,
        "device": device,
        "epochs": epochs,
        "split_sizes": {name: len(items) for name, items in split.items()},
        "test_accuracy": round(test["accuracy"], 4),
        "test_macro_f1": round(test["macro_f1"], 4),
        "classification_report": report,
        "note": "These metrics are meaningful only after an actual verified training run.",
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
