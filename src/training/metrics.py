"""The project's fixed test-time metric suite -- Cross-Entropy Loss, Top-1
Accuracy, Top-3 Accuracy, Balanced Accuracy, Macro-F1, Confusion Matrix (see
assets/docs/05_modelling_decisions.md, "Test-time metric suite") -- plus the
per-class breakdown they're derived from. Computed from plain Python lists
of true/predicted ids (no sklearn dependency -- keeps requirements.txt
unchanged). Per-class breakdown and the confusion matrix are zero-filled
against the FULL id space, never silently omitted for a class with no
support in this split -- same convention as src/data/stats.py's
class_counts() for the data stage.
"""
from __future__ import annotations

from collections import Counter


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Top-1 accuracy."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def top_k_accuracy(y_true: list[int], y_topk: list[list[int]]) -> float:
    """Top-k accuracy (k=3 everywhere in this project -- see
    engine.py:run_epoch, which builds y_topk from each head's top-3 logits).
    y_topk[i] is the list of top-k predicted ids for example i; correct if
    the true id is anywhere in that list. len(y_topk[i]) may be < k only
    when a head has fewer than k classes (not the case for crop/14 or
    disease/22, but handled rather than assumed)."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, topk in zip(y_true, y_topk) if t in topk)
    return correct / len(y_true)


def per_class_report(y_true: list[int], y_pred: list[int], id_to_name: dict) -> dict:
    """Returns {class_name: {support, correct, recall, precision, f1}} for
    EVERY id in id_to_name, including ones with zero support in y_true --
    those report support=0 and recall=None (never a misleading 0.0 or an
    omitted key)."""
    support = Counter(y_true)
    predicted_as = Counter(y_pred)
    correct = Counter(t for t, p in zip(y_true, y_pred) if t == p)

    report = {}
    for idx, name in id_to_name.items():
        n_support = support.get(idx, 0)
        n_correct = correct.get(idx, 0)
        n_predicted = predicted_as.get(idx, 0)
        recall = (n_correct / n_support) if n_support > 0 else None
        precision = (n_correct / n_predicted) if n_predicted > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
        report[name] = {
            "support": n_support,
            "correct": n_correct,
            "predicted_count": n_predicted,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return report


def macro_f1(per_class: dict) -> float:
    """Mean F1 over classes that actually had test support -- a class with
    support=0 has no recall to average in (see per_class_report), and is
    excluded here rather than treated as 0, matching the zero-fill-don't-
    silently-penalize convention."""
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    return sum(f1s) / len(f1s) if f1s else 0.0


def balanced_accuracy(per_class: dict) -> float:
    """Mean per-class recall over classes with nonzero support -- the
    standard balanced-accuracy definition (equal to sklearn.metrics.
    balanced_accuracy_score on the same y_true/y_pred), built from the same
    zero-filled per_class_report macro_f1 uses, for the same reason: a
    class with support=0 has no recall and is excluded, not scored as 0."""
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return sum(recalls) / len(recalls) if recalls else 0.0


def confusion_matrix(y_true: list[int], y_pred: list[int], id_to_name: dict) -> dict:
    """Full (rows=true, cols=predicted) confusion matrix, zero-filled
    against the FULL id space in id_to_name -- a class with zero test
    examples still gets an (all-zero) row rather than being dropped, same
    convention as per_class_report. `labels[i]` names row/column i;
    `matrix[i][j]` is the count of true-class-i examples predicted as
    class j (so `matrix[i][i]` is that class's correct count, and
    `sum(matrix[i])` is its support)."""
    ids = sorted(id_to_name.keys())
    index = {idx: i for i, idx in enumerate(ids)}
    n = len(ids)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return {"labels": [id_to_name[idx] for idx in ids], "matrix": matrix}
