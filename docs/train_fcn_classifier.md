# train_fcn_classifier.py

## Purpose

Trains an FCN-based 3-class age group classifier using pre-extracted ArcFace embeddings. This is a classification counterpart to `train_fcn.py`, trained from scratch (no regression checkpoint loaded).

Class-imbalance is handled via weighted CrossEntropyLoss with weights computed as `n_total / (3 * class_count)`.

## Model

`AgeClassifier` from `src/models.py`:
- Input: ArcFace embedding (default 512-dim)
- Hidden layers: Linear(512→256), BN, ReLU, Dropout(0.3), Linear(256→128), BN, ReLU, Dropout(0.2), Linear(128→64), ReLU
- Output: Linear(64→3) logits

## Inputs

- `dataset/embeddings/train_embeddings.npy` + `train_labels.npy`
- `dataset/embeddings/valid_embeddings.npy` + `valid_labels.npy`
- `dataset/embeddings/test_embeddings.npy`  + `test_labels.npy`
- `dataset/embeddings/embedding_dim.txt`

## Outputs

Saved to `results/fcn_classifier/`:

| File | Description |
|------|-------------|
| `predictions.csv` | true_class, pred_class, true_age, pred_prob_0/1/2 |
| `metrics.json` | accuracy, macro_f1, weighted_f1, per_class dict |
| `loss_curve.png` | Train loss + Val accuracy vs epoch |
| `confusion_matrix.png` | 3×3 heatmap |
| `checkpoints/best_model.pt` | Best checkpoint by val_accuracy |

## CLI Usage

```bash
python src/train_fcn_classifier.py \
    [--epochs 100] \
    [--lr 1e-3] \
    [--batch-size 64] \
    [--patience 15]
```
