# YOLO26n Baseline

This document describes the YOLO26n baseline model for parking space detection and occupancy classification.

## Task

The goal of the baseline is to detect parking spaces in parking lot images and classify each detected space as one of two classes:

| Class id | Class name |
|---:|---|
| 0 | `space-empty` |
| 1 | `space-occupied` |

This is a two-class object detection task. The model predicts:

- bounding boxes,
- class labels,
- confidence scores.

The background is not treated as a separate class. It is handled implicitly by YOLO.

## Dataset

The dataset is based on PKLot downloaded from Kaggle/Roboflow. The original annotations are in COCO format.

Raw dataset statistics:

| Split | Images | Annotations |
|---|---:|---:|
| Train | 8691 | 497856 |
| Validation | 2483 | 143316 |
| Test | 1242 | 70684 |

Original COCO categories:

| COCO category id | Name |
|---:|---|
| 0 | `spaces` |
| 1 | `space-empty` |
| 2 | `space-occupied` |

Category `0`, `spaces`, is a parent/placeholder category and is not used as a training class.

## Dataset inspection

During visual inspection, the bounding boxes were generally correct. However, some issues were found:

1. Some images had zero annotations despite visibly containing parking spaces.
2. Some `space-empty` / `space-occupied` labels appeared noisy.

Images with zero annotations:

| Split | Zero-annotation images | Percentage |
|---|---:|---:|
| Train | 189 / 8691 | 2.17% |
| Validation | 59 / 2483 | 2.38% |
| Test | 26 / 1242 | 2.09% |

Zero-annotation images are skipped during conversion to YOLO format. Keeping them would incorrectly teach the model that visible parking spaces are background.

## COCO to YOLO conversion

The dataset is converted from COCO format to YOLO format using:

```bash
uv run python scripts/convert_coco_to_yolo.py --overwrite
```

Class mapping:

| COCO category id | COCO class | YOLO class id | YOLO class |
|---:|---|---:|---|
| 1 | `space-empty` | 0 | `space-empty` |
| 2 | `space-occupied` | 1 | `space-occupied` |

YOLO label format:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0, 1]`.

Converted dataset statistics:

| Split | Images kept | Labels |
|---|---:|---:|
| Train | 8502 | 497856 |
| Validation | 2424 | 143316 |
| Test | 1216 | 70684 |

The converted dataset is saved to:

```text
data/yolo/
```

This directory is ignored by git.

## YOLO dataset structure

After conversion, the dataset has the following structure:

```text
data/yolo/
    images/
        train/
        valid/
        test/
    labels/
        train/
        valid/
        test/
    data.yaml
```

The `data.yaml` file:

```yaml
path: data/yolo
train: images/train
val: images/valid
test: images/test

names:
  0: "space-empty"
  1: "space-occupied"
```

## Visualizing annotations

COCO annotations can be visualized with:

```bash
uv run python scripts/visualize_coco_annotations.py \
  --split train \
  --num-images 8 \
  --seed 42
```

YOLO annotations after conversion can be visualized with:

```bash
uv run python scripts/visualize_yolo_annotations.py \
  --split train \
  --num-images 8 \
  --seed 42
```

Generated previews are saved under:

```text
outputs/dataset_preview/
```

## Model

The baseline model is:

```text
YOLO26n
```

Pretrained weights:

```text
yolo26n.pt
```

The model is fine-tuned for two classes:

- `space-empty`
- `space-occupied`

The weights file is downloaded automatically by Ultralytics if it is not present locally. Model weights are not committed to git.

## Smoke test

A small subset can be created for smoke tests:

```bash
uv run python scripts/create_yolo_subset.py \
  --output-dir data/yolo_smoke \
  --train-images 64 \
  --valid-images 16 \
  --test-images 16 \
  --seed 42 \
  --overwrite
```

Smoke test command:

```bash
uv run yolo detect train \
  model=yolo26n.pt \
  data=data/yolo_smoke/data.yaml \
  imgsz=640 \
  epochs=1 \
  batch=4 \
  device=mps \
  workers=0 \
  cache=False \
  val=False \
  name=smoke_yolo26n_pklot_small \
  exist_ok=True
```

## Mini-overfit test

A small overfit subset was used to check whether the model can learn the task:

```bash
uv run python scripts/create_yolo_subset.py \
  --output-dir data/yolo_overfit \
  --train-images 128 \
  --valid-images 32 \
  --test-images 32 \
  --seed 777 \
  --overwrite
```

Training command:

```bash
uv run yolo detect train \
  model=yolo26n.pt \
  data=data/yolo_overfit/data.yaml \
  imgsz=640 \
  epochs=30 \
  batch=16 \
  device=mps \
  workers=0 \
  cache=False \
  name=overfit_yolo26n_pklot_128 \
  exist_ok=True
```

The mini-overfit test confirmed that the model can learn the converted dataset.

Final mini-overfit metrics after 30 epochs:

| Metric | Value |
|---|---:|
| Precision | 0.780 |
| Recall | 0.774 |
| mAP@50 | 0.856 |
| mAP@50-95 | 0.529 |

## Final training

Final training command:

```bash
time uv run yolo detect train \
  model=yolo26n.pt \
  data=data/yolo/data.yaml \
  imgsz=768 \
  epochs=35 \
  batch=8 \
  device=mps \
  workers=0 \
  cache=False \
  patience=10 \
  save_period=5 \
  seed=42 \
  name=yolo26n_pklot_768_b8_e35 \
  exist_ok=True
```

Training setup:

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Image size | 768 |
| Batch size | 8 |
| Planned epochs | 35 |
| Completed fully validated epochs | 7 |
| Device | MPS, Apple M3 Max |
| Seed | 42 |

Training was interrupted with exit code `137`, which likely means that the process was killed due to memory pressure / out-of-memory. The final model uses the best checkpoint from the completed training run.

Final checkpoint:

```text
runs/detect/yolo26n_pklot_768_b8_e35/weights/best.pt
```

## Validation results

Best validation result was achieved at epoch 7:

| Metric | Value |
|---|---:|
| Precision | 0.96636 |
| Recall | 0.96952 |
| mAP@50 | 0.99195 |
| mAP@50-95 | 0.86523 |

## Test evaluation

Test evaluation command:

```bash
uv run yolo detect val \
  model=runs/detect/yolo26n_pklot_768_b8_e35/weights/best.pt \
  data=data/yolo/data.yaml \
  split=test \
  imgsz=768 \
  batch=8 \
  device=mps \
  workers=0 \
  name=yolo26n_pklot_768_b8_e35_test
```

Final test results:

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 1216 | 70684 | 0.967 | 0.969 | 0.991 | 0.866 |
| `space-empty` | 1041 | 36584 | 0.951 | 0.986 | 0.991 | 0.878 |
| `space-occupied` | 991 | 34100 | 0.982 | 0.951 | 0.992 | 0.854 |

Inference speed:

| Stage | Time per image |
|---|---:|
| Preprocess | 0.4 ms |
| Inference | 4.2 ms |
| Postprocess | 0.8 ms |

## Prediction visualization

Predictions can be generated with:

```bash
uv run yolo detect predict \
  model=runs/detect/yolo26n_pklot_768_b8_e35/weights/best.pt \
  source=data/yolo/images/test \
  imgsz=768 \
  conf=0.35 \
  device=mps \
  show_labels=False \
  show_conf=False \
  line_width=2 \
  name=yolo26n_pklot_768_b8_e35_test_predictions_clean \
  exist_ok=True
```

The resulting images are saved to:

```text
runs/detect/yolo26n_pklot_768_b8_e35_test_predictions_clean/
```

Representative predictions are copied to:

```text
outputs/final_results/predictions/
```

## Final artifacts

Final local artifacts are collected under:

```text
outputs/final_results/
```

Expected structure:

```text
outputs/final_results/
    training_results.csv
    yolo26n_test_metrics.txt
    checkpoints/
        yolo26n_pklot_best.pt
    plots/
        test_confusion_matrix.png
        test_confusion_matrix_normalized.png
        test_pr_curve.png
        test_f1_curve.png
        test_precision_curve.png
        test_recall_curve.png
        training_losses.png
        validation_metrics.png
    predictions/
        ...
```

These artifacts are not committed to git.

## Main observations

The YOLO26n baseline achieved strong results:

- `mAP@50 = 0.991`
- `mAP@50-95 = 0.866`
- precision and recall are both close to `0.97`

The model performs well for both classes. It has slightly higher recall for `space-empty` and slightly higher precision for `space-occupied`.

## Limitations

Known limitations:

1. Training did not complete all planned epochs because the process was killed with exit code `137`.
2. Some labels in the dataset appear noisy.
3. Images with zero annotations were removed during preprocessing.
4. No extensive hyperparameter search was performed.
5. The dataset contains repeated parking lot viewpoints, so train/validation/test scenes may be visually similar.

## Conclusion

The YOLO26n baseline successfully detects parking spaces and classifies them as empty or occupied.

Despite the interrupted training, the best checkpoint achieved strong test results and provides a solid baseline for future experiments.
