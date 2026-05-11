from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def copy_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    num_images: int,
    seed: int,
) -> None:
    source_images_dir = source_dir / "images" / split
    source_labels_dir = source_dir / "labels" / split

    output_images_dir = output_dir / "images" / split
    output_labels_dir = output_dir / "labels" / split

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in source_images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    random.seed(seed)
    selected_images = random.sample(image_paths, k=min(num_images, len(image_paths)))

    for image_path in selected_images:
        label_path = source_labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        shutil.copy2(image_path, output_images_dir / image_path.name)
        shutil.copy2(label_path, output_labels_dir / label_path.name)

    print(f"{split}: copied {len(selected_images)} images")


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "names:",
            '  0: "space-empty"',
            '  1: "space-occupied"',
            "",
        ]
    )

    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small YOLO dataset subset.")

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/yolo"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo_smoke"),
    )
    parser.add_argument(
        "--train-images",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--valid-images",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--test-images",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    copy_split(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split="train",
        num_images=args.train_images,
        seed=args.seed,
    )
    copy_split(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split="valid",
        num_images=args.valid_images,
        seed=args.seed,
    )
    copy_split(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split="test",
        num_images=args.test_images,
        seed=args.seed,
    )

    write_data_yaml(args.output_dir)

    print(f"Subset saved to: {args.output_dir}")
    print(f"Data YAML saved to: {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
