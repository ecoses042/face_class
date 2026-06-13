# train_small_cnn_classifier.py

## Purpose

Trains a Small CNN (~3.3M parameters) for 3-class age group classification directly on AIHub face images. This is a classification counterpart to `train_small_cnn_aihub.py`, trained from scratch.

Class-imbalance is handled via `class_weight` passed to `model.fit()`, computed with `compute_class_weights()` from `pipeline_aihub.py`.

## Model

`build_small_cnn_classifier` from `src/models_cnn.py`:
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(128) → Dropout(0.3) → Dense(3, softmax)

## Inputs

- AIHub face images via `load_splits(use_crops=False)` → uses `photo_age` column
- Split defined by `dataset/embeddings/split_meta.csv`

## Outputs

Saved to `results/cnn_small_classifier/`:

| File | Description |
|------|-------------|
| `predictions.csv` | true_class, pred_class, true_age |
| `metrics.json` | accuracy, macro_f1, weighted_f1, per_class dict |
| `loss_curve.png` | Train/val loss + accuracy vs epoch |
| `confusion_matrix.png` | 3×3 heatmap |
| `checkpoints/best_model.keras` | Best checkpoint by val_accuracy |

## CLI Usage

```bash
python src/train_small_cnn_classifier.py \
    [--img-size 128] \
    [--batch-size 32] \
    [--epochs 30] \
    [--patience 10] \
    [--lr 1e-3]
```
