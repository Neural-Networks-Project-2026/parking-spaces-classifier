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
        
    img_to_anns = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
        
    TARGET_W = 640
    TARGET_H = 360
        
    for img_info in coco["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        
        img_path = raw_dir / file_name
        if not img_path.exists():
            continue
            
        anns = img_to_anns[img_id]
        
        orig_width = img_info["width"]
        orig_height = img_info["height"]
        
        # Calculate scale and padding for letterbox
        scale = min(TARGET_W / orig_width, TARGET_H / orig_height)
        new_w = int(orig_width * scale)
        new_h = int(orig_height * scale)
        
        pad_x = (TARGET_W - new_w) // 2
        pad_y = (TARGET_H - new_h) // 2
        
        # Process image
        image = Image.open(img_path).convert("RGB")
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        final_image = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
        final_image.paste(resized_image, (pad_x, pad_y))
        final_image.save(processed_dir / file_name)
        
        img_info["width"] = TARGET_W
        img_info["height"] = TARGET_H
        
        if not anns:
            img_info["valid_area"] = [0, 0, TARGET_W, TARGET_H]
            continue
            
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            
            # Scale bounding boxes
            new_x = x * scale + pad_x
            new_y = y * scale + pad_y
            new_box_w = w * scale
            new_box_h = h * scale
            
            ann["bbox"] = [new_x, new_y, new_box_w, new_box_h]
            
            # Find bounds for valid_area
            min_x = min(min_x, new_x)
            min_y = min(min_y, new_y)
            max_x = max(max_x, new_x + new_box_w)
            max_y = max(max_y, new_y + new_box_h)
            
        scaled_padding = padding * scale
        valid_min_x = max(0, min_x - scaled_padding)
        valid_min_y = max(0, min_y - scaled_padding)
        valid_max_x = min(TARGET_W, max_x + scaled_padding)
        valid_max_y = min(TARGET_H, max_y + scaled_padding)
        
        img_info["valid_area"] = [valid_min_x, valid_min_y, valid_max_x, valid_max_y]
            
    out_ann_file = processed_dir / "_annotations.coco.json"
    with open(out_ann_file, "w") as f:
        json.dump(coco, f, indent=4)
        
    print(f"Finished processing {subset_name}. Images saved to {processed_dir}")

if __name__ == "__main__":
    for subset in ["train", "valid", "test"]:
        process_subset(subset, padding=20)
