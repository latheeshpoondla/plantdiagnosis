"""Train/eval loops shared by every (backbone, regime, seed) combination --
scripts/train.py just wires these up with the right dataloaders/model/loss
weights. Mixed precision only activates on CUDA (amp is a no-op on CPU --
there's nothing to gain and it avoids autocast/GradScaler edge cases on a
device that will never actually be used for a real training run).
"""
from __future__ import annotations

from src.training.metrics import (
    accuracy, per_class_report, macro_f1, top_k_accuracy, balanced_accuracy, confusion_matrix,
)

TOP_K = 3  # the project's fixed "Top-3 Accuracy" metric -- see assets/docs/05_modelling_decisions.md


def _unpack_labels(labels, device):
    # multitask labels arrive as a (crop_id_tensor, disease_id_tensor) tuple,
    # per src/data/dataset.py's PlantLeafDataset + torch's default_collate.
    crop_ids, disease_ids = labels
    return crop_ids.to(device, non_blocking=True), disease_ids.to(device, non_blocking=True)


def run_epoch(
    model,
    loader,
    device,
    crop_loss_fn,
    disease_loss_fn,
    loss_weight_crop: float,
    loss_weight_disease: float,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    collect_predictions: bool = False,
) -> dict:
    """One pass over `loader`. Pass `optimizer` (and `scaler` if amp) for a
    training epoch; omit both for a no-grad eval pass. Returns a metrics
    dict; if collect_predictions, also includes 'y_true_crop'/'y_pred_crop'/
    'y_true_disease'/'y_pred_disease' (top-1) and 'y_pred_crop_top3'/
    'y_pred_disease_top3' lists -- used to build the full test-time metric
    suite (build_test_report), not computed on every val epoch since the
    per-example lists aren't needed for early stopping.
    """
    import torch

    is_train = optimizer is not None
    model.train(is_train)
    use_amp = amp and is_train and device.type == "cuda"

    total_loss = total_crop_loss = total_disease_loss = 0.0
    n = 0
    y_true_crop, y_pred_crop = [], []
    y_true_disease, y_pred_disease = [], []
    y_pred_crop_top3, y_pred_disease_top3 = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            crop_ids, disease_ids = _unpack_labels(labels, device)
            bs = images.size(0)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                crop_logits, disease_logits = model(images)
                crop_loss = crop_loss_fn(crop_logits, crop_ids)
                disease_loss = disease_loss_fn(disease_logits, disease_ids)
                loss = loss_weight_crop * crop_loss + loss_weight_disease * disease_loss

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * bs
            total_crop_loss += crop_loss.item() * bs
            total_disease_loss += disease_loss.item() * bs
            n += bs

            y_true_crop.extend(crop_ids.detach().cpu().tolist())
            y_pred_crop.extend(crop_logits.detach().argmax(dim=1).cpu().tolist())
            y_true_disease.extend(disease_ids.detach().cpu().tolist())
            y_pred_disease.extend(disease_logits.detach().argmax(dim=1).cpu().tolist())

            if collect_predictions:
                k_crop = min(TOP_K, crop_logits.size(1))
                k_disease = min(TOP_K, disease_logits.size(1))
                y_pred_crop_top3.extend(crop_logits.detach().topk(k_crop, dim=1).indices.cpu().tolist())
                y_pred_disease_top3.extend(disease_logits.detach().topk(k_disease, dim=1).indices.cpu().tolist())

    crop_acc = accuracy(y_true_crop, y_pred_crop)
    disease_acc = accuracy(y_true_disease, y_pred_disease)
    result = {
        "loss": total_loss / n,
        "crop_loss": total_crop_loss / n,
        "disease_loss": total_disease_loss / n,
        "crop_acc": crop_acc,
        "disease_acc": disease_acc,
        "mean_acc": (crop_acc + disease_acc) / 2,
        "n_examples": n,
    }
    if collect_predictions:
        result.update({
            "y_true_crop": y_true_crop, "y_pred_crop": y_pred_crop,
            "y_true_disease": y_true_disease, "y_pred_disease": y_pred_disease,
            "y_pred_crop_top3": y_pred_crop_top3, "y_pred_disease_top3": y_pred_disease_top3,
        })
    return result


def _head_report(y_true, y_pred, y_pred_top3, loss, id_to_name) -> dict:
    """The project's fixed test-time metric suite for one head (crop or
    disease): Cross-Entropy Loss, Top-1 Accuracy, Top-3 Accuracy, Balanced
    Accuracy, Macro-F1, Confusion Matrix -- plus the per-class breakdown
    those are derived from. See assets/docs/05_modelling_decisions.md,
    "Test-time metric suite"."""
    per_class = per_class_report(y_true, y_pred, id_to_name)
    return {
        "cross_entropy_loss": loss,
        "top1_accuracy": accuracy(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),  # alias, kept for backward compat
        "top3_accuracy": top_k_accuracy(y_true, y_pred_top3),
        "balanced_accuracy": balanced_accuracy(per_class),
        "macro_f1": macro_f1(per_class),
        "confusion_matrix": confusion_matrix(y_true, y_pred, id_to_name),
        "per_class": per_class,
    }


def build_test_report(epoch_result: dict, taxonomy) -> dict:
    """Turns a collect_predictions=True run_epoch() result into the full
    test_metrics.json payload: the fixed metric suite (Cross-Entropy Loss,
    Top-1/Top-3 Accuracy, Balanced Accuracy, Macro-F1, Confusion Matrix) for
    each head, plus mean-of-both-heads summaries at the top level."""
    crop = _head_report(epoch_result["y_true_crop"], epoch_result["y_pred_crop"],
                         epoch_result["y_pred_crop_top3"], epoch_result["crop_loss"], taxonomy.id_to_crop)
    disease = _head_report(epoch_result["y_true_disease"], epoch_result["y_pred_disease"],
                            epoch_result["y_pred_disease_top3"], epoch_result["disease_loss"], taxonomy.id_to_disease)
    return {
        "n_examples": epoch_result["n_examples"],
        "cross_entropy_loss": epoch_result["loss"],
        "mean_top1_accuracy": (crop["top1_accuracy"] + disease["top1_accuracy"]) / 2,
        "mean_top3_accuracy": (crop["top3_accuracy"] + disease["top3_accuracy"]) / 2,
        "mean_balanced_accuracy": (crop["balanced_accuracy"] + disease["balanced_accuracy"]) / 2,
        "mean_macro_f1": (crop["macro_f1"] + disease["macro_f1"]) / 2,
        "crop": crop,
        "disease": disease,
    }
