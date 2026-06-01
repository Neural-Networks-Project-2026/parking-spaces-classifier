import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import v2
from torchvision import tv_tensors

from src.models.custom_detector import create_centernet_targets


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
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(image.height, image.width)
        ) if boxes else tv_tensors.BoundingBoxes(
            torch.empty((0, 4), dtype=torch.float32), format="XYXY", canvas_size=(image.height, image.width)
        )
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)
            
        _, H, W = image.shape
        hm, wh, offset, reg_mask = create_centernet_targets(
            target["boxes"], target["labels"], H, W, torch.device("cpu")
        )
        
        target["heatmap"] = hm
        target["wh"] = wh
        target["offset"] = offset
        target["reg_mask"] = reg_mask

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch, strict=False))

def get_transform(is_train=True):
    transforms = [
        v2.ToImage(),
        v2.Resize(size=(360, 640), antialias=True),
    ]
    if is_train:
        transforms.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(transforms)


# if __name__ == "__main__":
#     train_img_dir = "data/raw/train"
#     train_ann_file = "data/raw/train/_annotations.coco.json"

#     train_dataset = PKLotDataset(
#         root_dir=train_img_dir,
#         annotation_file=train_ann_file,
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=4,
#         shuffle=True,
#         num_workers=2,
#         collate_fn=collate_fn,
#     )

#     for images, targets in train_loader:
#         print(f"Kształt zdjęcia: {images[0].shape}")
#         print(f"Ilość obiektów na pierwszym zdjęciu: {len(targets[0]['labels'])}")
#         break
