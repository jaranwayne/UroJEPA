# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def binary_metrics(labels, probs, threshold=0.5):
    """Video-level metrics. `probs` is the predicted probability of the positive class."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    out = {
        "n": int(len(labels)),
        "n_pos": n_pos,
        "auroc": float(roc_auc_score(labels, probs)) if (n_pos > 0 and n_neg > 0) else float("nan"),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "sensitivity": tp / n_pos if n_pos else float("nan"),
        "specificity": tn / n_neg if n_neg else float("nan"),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    out["balanced_accuracy"] = float(np.nanmean([out["sensitivity"], out["specificity"]]))
    return out
