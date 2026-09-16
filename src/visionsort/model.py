from __future__ import annotations

from pathlib import Path

from PIL import Image

from .data import ImageRecord, LABELS


def build_transforms(train: bool = False):
    import torchvision.transforms as T

    steps = [T.Resize((224, 224))]
    if train:
        steps.extend([T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.15, contrast=0.15)])
    steps.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(steps)


class WasteDataset:
    def __init__(self, records: list[ImageRecord], train: bool = False):
        from torch.utils.data import Dataset

        if not issubclass(type(self), Dataset):
            # This object intentionally implements the Dataset protocol without importing torch at module load.
            pass
        self.records = records
        self.transform = build_transforms(train=train)
        self.label_to_id = {label: idx for idx, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_id[record.label]


def build_small_cnn(num_classes: int = len(LABELS)):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, num_classes),
    )


def build_mobilenet(num_classes: int = len(LABELS), pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def load_weights(model, weights_path: str | Path, device: str = "cpu"):
    import torch

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
