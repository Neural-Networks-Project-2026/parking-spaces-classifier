from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLASS_MAPPING = {
    1: 0,  # space-empty -> space-empty
    2: 1,  # space-occupied -> space-occupied
}

YOLO_NAMES = {
    0: "space-empty",
    1: "space-occupied",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    for split in ["train", "valid", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_bbox_to_yolo(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    x, y, width, height = bbox

    if width <= 0 or height <= 0:
        return None

    x_center = x + width / 2
    y_center = y + height / 2

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height

    values = (x_center_norm, y_center_norm, width_norm, height_norm)

    if any(value < 0 or value > 1 for value in values):
        return None

    return values


def copy_image(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def convert_split(
    raw_data_dir: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    split_dir = raw_data_dir / split
    annotation_path = split_dir / "_annotations.coco.json"

    coco = load_json(annotation_path)

    images = coco["images"]
    annotations = coco["annotations"]

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    stats: dict[str, Any] = {
        "split": split,
        "total_images": len(images),
        "kept_images": 0,
        "skipped_zero_annotation_images": 0,
        "skipped_no_usable_labels_images": 0,
        "missing_images": 0,
        "invalid_boxes": 0,
        "total_labels": 0,
        "class_counts": Counter(),
        "skipped_files": [],
        "missing_files": [],
    }

    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        image_annotations = annotations_by_image_id.get(image_id, [])

        if not image_annotations:
            stats["skipped_zero_annotation_images"] += 1
            stats["skipped_files"].append(file_name)
            continue

        yolo_lines = []

        for annotation in image_annotations:
            coco_category_id = annotation["category_id"]

            if coco_category_id not in CLASS_MAPPING:
                continue

            yolo_class_id = CLASS_MAPPING[coco_category_id]
            yolo_bbox = convert_bbox_to_yolo(
                bbox=annotation["bbox"],
                image_width=image_width,
                image_height=image_height,
            )

            if yolo_bbox is None:
                stats["invalid_boxes"] += 1
                continue

            x_center, y_center, width, height = yolo_bbox

            yolo_lines.append(
                f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            stats["class_counts"][yolo_class_id] += 1
            stats["total_labels"] += 1

        if not yolo_lines:
            stats["skipped_no_usable_labels_images"] += 1
            stats["skipped_files"].append(file_name)
            continue

        source_image_path = split_dir / file_name
        target_image_path = output_dir / "images" / split / file_name
        target_label_path = output_dir / "labels" / split / f"{Path(file_name).stem}.txt"

        if not source_image_path.exists():
            stats["missing_images"] += 1
            stats["missing_files"].append(str(source_image_path))
            continue

        copy_image(source_image_path, target_image_path)

        with target_label_path.open("w", encoding="utf-8") as label_file:
            label_file.write("\n".join(yolo_lines))
            label_file.write("\n")

        stats["kept_images"] += 1

    stats["class_counts"] = dict(stats["class_counts"])

    return stats


def write_data_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"

    content = "\n".join(
        [
            "path: data/yolo",
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

    yaml_path.write_text(content, encoding="utf-8")


def write_summary(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "conversion_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    for split_summary in summary:
        split = split_summary["split"]
        skipped_path = output_dir / f"skipped_{split}.txt"
        skipped_files = split_summary["skipped_files"]

        skipped_path.write_text(
            "\n".join(skipped_files) + ("\n" if skipped_files else ""),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PKLot COCO annotations to YOLO format.")

    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw COCO dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo"),
        help="Path where YOLO dataset will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directory before conversion.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_output_dirs(output_dir=args.output_dir, overwrite=args.overwrite)

    summary = []

    for split in ["train", "valid", "test"]:
        split_summary = convert_split(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            split=split,
        )
        summary.append(split_summary)

        print("=" * 80)
        print(f"SPLIT: {split}")
        print(f"Total images: {split_summary['total_images']}")
        print(f"Kept images: {split_summary['kept_images']}")
        print(f"Skipped zero-annotation images: {split_summary['skipped_zero_annotation_images']}")
        print(f"Skipped no-usable-label images: {split_summary['skipped_no_usable_labels_images']}")
        print(f"Missing images: {split_summary['missing_images']}")
        print(f"Invalid boxes: {split_summary['invalid_boxes']}")
        print(f"Total labels: {split_summary['total_labels']}")
        print(f"Class counts: {split_summary['class_counts']}")

    write_data_yaml(output_dir=args.output_dir)
    write_summary(output_dir=args.output_dir, summary=summary)

    print("=" * 80)
    print(f"YOLO dataset saved to: {args.output_dir}")
    print(f"Data YAML saved to: {args.output_dir / 'data.yaml'}")
    print(f"Conversion summary saved to: {args.output_dir / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()
