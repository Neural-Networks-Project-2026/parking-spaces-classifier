from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: "empty",
    1: "occupied",
}

CLASS_COLORS = {
    0: "lime",
    1: "red",
}


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []

    labels = []

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append((class_id, x_center, y_center, width, height))

    return labels


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    box_width = width * image_width
    box_height = height * image_height

    center_x = x_center * image_width
    center_y = y_center * image_height

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    return x1, y1, x2, y2


def draw_labels(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    show_labels: bool,
) -> None:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("Arial.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    labels = read_yolo_labels(label_path)

    for class_id, x_center, y_center, width, height in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )

        color = CLASS_COLORS[class_id]
        class_name = CLASS_NAMES[class_id]

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2,
        )

        if show_labels:
            draw.text(
                (x1, max(0, y1 - 14)),
                class_name,
                fill=color,
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Saved {output_path} with {len(labels)} labels")


def visualize_split(
    yolo_data_dir: Path,
    output_dir: Path,
    split: str,
    num_images: int,
    seed: int,
    show_labels: bool,
) -> None:
    images_dir = yolo_data_dir / "images" / split
    labels_dir = yolo_data_dir / "labels" / split

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    random.seed(seed)
    sampled_image_paths = random.sample(image_paths, k=min(num_images, len(image_paths)))

    for image_path in sampled_image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        output_path = output_dir / split / image_path.name

        draw_labels(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            show_labels=show_labels,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels after COCO to YOLO conversion."
    )

    parser.add_argument(
        "--yolo-data-dir",
        type=Path,
        default=Path("data/yolo"),
        help="Path to YOLO dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_preview/yolo_gt"),
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
        yolo_data_dir=args.yolo_data_dir,
        output_dir=args.output_dir,
        split=args.split,
        num_images=args.num_images,
        seed=args.seed,
        show_labels=args.show_labels,
    )


if __name__ == "__main__":
    main()
