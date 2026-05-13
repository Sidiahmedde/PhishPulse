# Super Model Fusion Results

## Models and Outputs

- Fusion model: `src/models/fusion_model_super_eval.pkl`
- Sender model: `src/models/sender_model_super_v2.pkl`
- Sender vectorizer: `src/models/sender_vectorizer_super_v2.pkl`
- Graphs folder: `graphs/`

Graphs generated:

- `graphs/graph_fusion_confusion_matrix.png`
- `graphs/graph_fusion_roc_curve.png`
- `graphs/graph_fusion_precision_recall_curve.png`

## Final Fusion Model Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9931 |
| Precision | 0.9897 |
| Recall | 0.9965 |
| F1-score | 0.9931 |
| ROC-AUC | 0.9997 |
| PR-AUC / Average Precision | 0.9997 |

## Confusion Matrix

| Actual \ Predicted | Legitimate (`0`) | Phishing (`1`) |
|---|---:|---:|
| Legitimate (`0`) | 3423 | 36 |
| Phishing (`1`) | 12 | 3447 |

## Proposal-Aligned Graphs

### ROC Curve

- File: `graphs/graph_fusion_roc_curve.png`
- ROC-AUC: `0.9997`

### Precision-Recall Curve

- File: `graphs/graph_fusion_precision_recall_curve.png`
- Average Precision: `0.9997`

### Confusion Matrix Figure

- File: `graphs/graph_fusion_confusion_matrix.png`

## Learned Meta-Model Weights

The fusion model uses learned coefficients over module outputs and helper features:

| Feature | Coefficient | Odds Ratio |
|---|---:|---:|
| `p_header` | 13.9394 | 1131848.6247 |
| `p_sender` | 11.0825 | 65021.5661 |
| `p_subject` | 5.9117 | 369.3444 |
| `p_body` | 4.4715 | 87.4899 |
| `has_url` | 0.6225 | 1.8636 |
| `url_count` | -0.1164 | 0.8901 |
| `p_url` | -0.8006 | 0.4491 |

Interpretation:

- `p_header` and `p_sender` contribute the strongest positive evidence.
- `p_subject` and `p_body` also contribute strongly.
- URL-related auxiliary features contribute less than the sender/header/body/subject signals on this dataset.

## Baseline Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Best single module (`sender`) | 0.9426 | 0.9595 | 0.9243 | 0.9415 | 0.9453 | 0.9686 |
| Equal-weight average fusion | 0.7998 | 1.0000 | 0.5996 | 0.7497 | 0.9965 | 0.9966 |
| Meta-model fusion | 0.9931 | 0.9897 | 0.9965 | 0.9931 | 0.9997 | 0.9997 |

Conclusion:

- The learned meta-model clearly outperforms the best single module.
- The learned meta-model also strongly outperforms simple equal-weight averaging, which justifies using trained fusion instead of fixed weights.

## Ablation Study

| Removed Feature | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `header` | 0.8141 | 0.7473 | 0.9491 | 0.8362 | 0.9496 | 0.9705 |
| `sender` | 0.9734 | 0.9748 | 0.9720 | 0.9734 | 0.9965 | 0.9968 |
| `subject` | 0.9824 | 0.9771 | 0.9879 | 0.9825 | 0.9984 | 0.9983 |
| `body` | 0.9918 | 0.9868 | 0.9968 | 0.9918 | 0.9997 | 0.9997 |
| `url` | 0.9928 | 0.9908 | 0.9948 | 0.9928 | 0.9997 | 0.9997 |

Interpretation:

- Removing `header` causes the largest performance drop, making it the most critical module in this setup.
- Removing `sender` also causes a substantial drop, confirming the value of the sender-only model.
- `subject` contributes meaningfully, while `body` and `url` provide smaller but still positive gains.

## Files for Report Use

- Metrics table: `dataset/artifacts/super_eval/fusion_metrics.csv`
- Confusion matrix values: `dataset/artifacts/super_eval/fusion_confusion_matrix.csv`
- Learned coefficients: `dataset/artifacts/super_eval/fusion_coefficients.csv`
- Baseline comparison: `dataset/artifacts/super_eval/fusion_baselines.csv`
- Ablation study: `dataset/artifacts/super_eval/fusion_ablation.csv`
