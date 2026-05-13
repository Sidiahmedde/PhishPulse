# Phishing Detector — Body Model

This repo contains a phishing detector for **email body text** using TF‑IDF,
handcrafted numeric features, and Logistic Regression.

## Dataset

We use CEAS_08 plus EPVME (malicious EMLs) and build a cleaned, body‑only CSV.
The training pipeline expects at least `body` and `label` columns.

Cleaned combined dataset (recommended):
`dataset/processed/body_combined_clean.csv`

## Train the Model

From repo root:

```bash
python -m src.fusion.train_body_model \
  --data_path dataset/processed/body_combined_clean.csv \
  --out_dir dataset/artifacts/body_best \
  --seed 42 \
  --max_features 30000 \
  --ngram_range 1 2 \
  --min_df 2 \
  --max_df 1.0 \
  --class_weight balanced
```

Artifacts will be saved to `dataset/artifacts/body_best/`:

- `body_model.joblib`
- `body_metrics.json`
- `body_roc.png`
- `body_pr.png`
- `splits.npz`

## Classify a Single Email Body (CLI)

```bash
# Text inline
python -m src.fusion.predict_body --model_path dataset/artifacts/body_best/body_model.joblib \
  --text "Paste your email body here"

# From file
python -m src.fusion.predict_body --model_path dataset/artifacts/body_best/body_model.joblib \
  --file path/to/email.txt

# From stdin
cat path/to/email.txt | python -m src.fusion.predict_body \
  --model_path dataset/artifacts/body_best/body_model.joblib
```

Recommended decision threshold (tuned on validation): `0.15`

You can adjust the decision threshold:

```bash
python -m src.fusion.predict_body --model_path dataset/artifacts/body_best/body_model.joblib \
  --text "..." --threshold 0.15
```

## Tests

Using the local venv:

```bash
.venv/bin/pytest tests/test_body_model.py -v
```
