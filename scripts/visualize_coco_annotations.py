from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CATEGORY_NAMES = {
    1: "empty",
    2: "occupied",
}

CATEGORY_COLORS = {
    1: "lime",
    2: "red",
}


def load_coco(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def group_annotations_by_image_id(annotations: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)

    for annotation in annotations:
        grouped[annotation["image_id"]].append(annotation)

    return grouped


def draw_annotations(
    image_path: Path,
    annotations: list[dict],
    output_path: Path,
    show_labels: bool,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("Arial.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    for annotation in annotations:
        category_id = annotation["category_id"]

        if category_id not in CATEGORY_NAMES:
            continue

        x, y, width, height = annotation["bbox"]

        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        color = CATEGORY_COLORS[category_id]
        label = CATEGORY_NAMES[category_id]

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2,
        )

        if show_labels:
            text_position = (x1, max(0, y1 - 14))
            draw.text(text_position, label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def visualize_split(
    raw_data_dir: Path,
    output_dir: Path,
    split: str,
    num_images: int,
    seed: int,
    show_labels: bool,
) -> None:
    split_dir = raw_data_dir / split
    annotation_path = split_dir / "_annotations.coco.json"

    coco = load_coco(annotation_path)

    images = coco["images"]
    annotations = coco["annotations"]

    annotations_by_image_id = group_annotations_by_image_id(annotations)

    random.seed(seed)
    sampled_images = random.sample(images, k=min(num_images, len(images)))

    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    for image_info in sampled_images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = split_dir / file_name
        output_path = split_output_dir / file_name

        image_annotations = annotations_by_image_id.get(image_id, [])

        draw_annotations(
            image_path=image_path,
            annotations=image_annotations,
            output_path=output_path,
            show_labels=show_labels,
        )

        empty_count = sum(1 for ann in image_annotations if ann["category_id"] == 1)
        occupied_count = sum(1 for ann in image_annotations if ann["category_id"] == 2)

        print(
            f"Saved {output_path} "
            f"with {len(image_annotations)} annotations "
            f"(empty={empty_count}, occupied={occupied_count})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize COCO ground-truth annotations for PKLot."
    )

    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_preview/coco_gt"),
        help="Path where visualized images will be saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default="train",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of random images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show class names next to boxes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    visualize_split(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        split=args.split,
        num_images=args.num_images,
        seed=args.seed,
        show_labels=args.show_labels,
    )


if __name__ == "__main__":
    main()
