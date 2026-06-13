# analyze_regression_to_cls.py

## Purpose

Converts existing regression predictions to a 3-class age group classification problem and computes performance metrics. This allows comparing how well each regression model implicitly separates the three age groups without any classification-specific training.

## Class Definitions

| Class | Label | Age Range |
|-------|-------|-----------|
| 0 | young | age < 20 |
| 1 | adult | 20 ≤ age < 60 |
| 2 | senior | age ≥ 60 |

## Inputs

Reads `predictions.csv` from the following result directories:

- `results/fcn_regressor/predictions.csv`
- `results/cnn_small/predictions.csv`
- `results/cnn_vgg/predictions.csv`

Each file must contain columns: `true_age`, `pred_age`, `error`.

Missing files are skipped with a warning.

## Outputs

All outputs are saved to `results/classification_analysis/regression_cls/`:

| File | Description |
|------|-------------|
| `metrics_{model_name}.json` | accuracy, macro_f1, weighted_f1, per_class dict |
| `confusion_matrix_{model_name}.png` | Heatmap with class labels young/adult/senior |
| `comparison_table.csv` | Model × metric comparison table |

## CLI Usage

```bash
python src/analyze_regression_to_cls.py
```

No command-line arguments. Reads all available prediction files automatically.
