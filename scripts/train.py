from __future__ import annotations

import argparse

from visionsort.data import download_trashnet
from visionsort.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VisionSort AI")
    parser.add_argument("--architecture", choices=["cnn", "mobilenet"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    root = download_trashnet(args.data_dir)
    summary = train_model(
        root,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.artifacts,
    )
    print(summary)


if __name__ == "__main__":
    main()
