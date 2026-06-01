import os
import json
from pathlib import Path
from PIL import Image

def process_subset(subset_name: str, padding: int = 20):
    raw_dir = Path(f"data/raw/{subset_name}")
    processed_dir = Path(f"data/processed/{subset_name}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    ann_file = raw_dir / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"Skipping {subset_name}, annotation file not found.")
        return
        
    print(f"Processing {subset_name} dataset...")
    
    with open(ann_file) as f:
        coco = json.load(f)
        
    # Build a lookup for annotations by image_id
    img_to_anns = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
        
    for img_info in coco["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        
        img_path = raw_dir / file_name
        if not img_path.exists():
            continue
            
        anns = img_to_anns[img_id]
        
        # If no annotations, we can't determine crop boundary, just copy the whole image
        if not anns:
            image = Image.open(img_path).convert("RGB")
            image.save(processed_dir / file_name)
            continue
            
        # Find global min/max coordinates
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            
        # Add padding
        orig_width = img_info["width"]
        orig_height = img_info["height"]
        
        crop_min_x = max(0, int(min_x - padding))
        crop_min_y = max(0, int(min_y - padding))
        crop_max_x = min(orig_width, int(max_x + padding))
        crop_max_y = min(orig_height, int(max_y + padding))
        
        # Open and crop image
        image = Image.open(img_path).convert("RGB")
        cropped_image = image.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))
        cropped_image.save(processed_dir / file_name)
        
        # Update image info
        img_info["width"] = cropped_image.width
        img_info["height"] = cropped_image.height
        
        # Update bounding boxes
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Shift coordinates relative to the new crop origin
            new_x = max(0, x - crop_min_x)
            new_y = max(0, y - crop_min_y)
            ann["bbox"] = [new_x, new_y, w, h]
            
    # Save the updated annotations
    out_ann_file = processed_dir / "_annotations.coco.json"
    with open(out_ann_file, "w") as f:
        json.dump(coco, f, indent=4)
        
    print(f"Finished processing {subset_name}. Images saved to {processed_dir}")

if __name__ == "__main__":
    for subset in ["train", "valid", "test"]:
        process_subset(subset, padding=20)
