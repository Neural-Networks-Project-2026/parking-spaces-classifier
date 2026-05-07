import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


class PKLotDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        with open(annotation_file) as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.img_to_anns = {img["id"]: [] for img in self.coco["images"]}

        for ann in self.coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.img_to_anns[img_id]
        boxes = []
        labels = []

        for ann in anns:
            x_min, y_min, w, h = ann["bbox"]
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(ann["category_id"])

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target


# Funkcja zapobiegająca błędowi sklejania tensorów o różnym rozmiarze
def collate_fn(batch):
    return tuple(zip(*batch, strict=False))


if __name__ == "__main__":
    train_img_dir = "data/raw/train"
    train_ann_file = "data/raw/train/_annotations.coco.json"

    train_dataset = PKLotDataset(
        root_dir=train_img_dir,
        annotation_file=train_ann_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    for images, targets in train_loader:
        print(f"Kształt zdjęcia: {images[0].shape}")
        print(f"Ilość obiektów na pierwszym zdjęciu: {len(targets[0]['labels'])}")
        break
