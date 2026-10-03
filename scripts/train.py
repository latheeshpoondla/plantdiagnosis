#!/usr/bin/env python3
"""Single entrypoint for every (backbone, regime, seed) training run of the
multitask (crop head + disease head) formulation. See
assets/docs/05_modelling_decisions.md for why this formulation/these
backbones/this regime order were chosen, and assets/docs/06_training_guide.md
for the full CLI reference and example commands.

Everything that changes between runs is a flag -- regime 3 is regime 2's
exact command with --regime 3 swapped in (it auto-loads the matching
regime-1 checkpoint); a different backbone is one flag; a from-scratch
zero-shot check of a regime-1 model on PlantDoc is --mode test plus two
flags on the SAME command used to train it.

    # regime 1, ConvNeXt-Tiny, default everything else
    python scripts/train.py --backbone convnext_tiny --regime 1

    # regime 1, Swin-Tiny
    python scripts/train.py --backbone swin_tiny --regime 1

    # regime 3 (auto-loads this backbone's regime-1 checkpoint), one of 3 seeds
    python scripts/train.py --backbone convnext_tiny --regime 3 --seed 42

    # zero-shot: evaluate a trained regime-1 model directly on PlantDoc, no
    # fine-tuning -- the "free" eval from the 2026-09-19 staged plan. For
    # regime 1 specifically, --eval_split defaults to "all" (PlantDoc's
    # complete train+val+test, not just its test split) since regime 1
    # never trained/validated on any of it -- pass --eval_split explicitly
    # to look at just one split instead.
    python scripts/train.py --backbone convnext_tiny --regime 1 --seed 42 \\
        --mode test --eval_dataset plantdoc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.taxonomy import load_taxonomy
from src.data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.training.class_weights import compute_class_weights
from src.training.experiment import experiment_dir, resolve_pretrained_source, REGIME_DATASET
from src.training.checkpoint import save_checkpoint, load_checkpoint, load_backbone_weights_only
from src.training.engine import run_epoch, build_test_report
from src.utils.io_utils import read_json, write_json, ensure_dir
from src.utils.seed_utils import set_global_seed
from src.utils.logging_utils import RunLogger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["convnext_tiny", "swin_tiny"])
    p.add_argument("--regime", required=True, type=int, choices=[1, 2, 3])
    p.add_argument("--label_mode", default="multitask", choices=["multitask"],
                   help="Only multitask is implemented so far -- see assets/docs/05_modelling_decisions.md.")
    p.add_argument("--mode", default="train", choices=["train", "test"],
                   help="'train': full train+val-per-epoch+test-on-best-epoch. "
                        "'test': evaluate an existing checkpoint only (e.g. a regime-1 "
                        "model's zero-shot check on PlantDoc) -- no training happens.")
    p.add_argument("--seed", type=int, default=None, help="Default: this regime's policy seed (see configs/model_config.json).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--optimizer", default=None, choices=["adamw", "sgd"])
    p.add_argument("--lr_schedule", default=None, choices=["cosine", "none"])
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--class_weight_mode", default=None, choices=["none", "inv_freq", "sqrt_inv_freq"])
    p.add_argument("--norm_stats", default=None, choices=["auto", "imagenet", "plantvillage", "plantdoc"],
                   help="'auto' (default): use the real channel mean/std computed for whichever "
                        "dataset is being trained/evaluated on (see 05_modelling_decisions.md).")
    p.add_argument("--pretrained_source", default=None, choices=["auto", "imagenet", "pvd_regime1", "none"])
    p.add_argument("--val_metric", default=None, choices=["mean_acc", "loss", "crop_acc", "disease_acc"],
                   help="Metric used to pick the best epoch's checkpoint (the one --mode test "
                        "later evaluates). 'loss' is minimized; everything else is maximized.")
    p.add_argument("--patience", type=int, default=None, help="Early-stop after this many epochs with no val_metric improvement. 0 disables.")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--loss_weight_crop", type=float, default=None)
    p.add_argument("--loss_weight_disease", type=float, default=None)

    p.add_argument("--manifests_dir", default=str(REPO_ROOT / "data_manifests"))
    p.add_argument("--configs_dir", default=str(REPO_ROOT / "configs"))
    p.add_argument("--runs_dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dry_run", action="store_true", help="2 batches, 1 epoch, no test -- smoke-test the pipeline, not a real run.")

    # --mode test only
    p.add_argument("--checkpoint", default=None, help="--mode test only. Defaults to this run's own best.pt.")
    p.add_argument("--eval_dataset", default=None, choices=["plantvillage", "plantdoc"],
                   help="--mode test only. Defaults to this regime's own dataset (no domain switch).")
    p.add_argument("--eval_split", default=None, choices=["train", "val", "test", "all"],
                   help="--mode test only. Default: 'all' (train+val+test combined) for regime 1's "
                        "cross-dataset PlantDoc eval specifically -- regime 1 never trained or "
                        "validated on any PlantDoc split, so none of it needs holding out, and "
                        "using all of it maximizes N for PlantDoc's thin per-class support. "
                        "'test' otherwise. Pass this explicitly to override either default.")

    return p.parse_args()


def resolve_config(args, model_cfg: dict) -> dict:
    """Fills every unset CLI flag from configs/model_config.json, resolved
    per-regime (heavy vs light) where relevant. Returns the full resolved
    dict that gets written verbatim to this run's config.json."""
    regime_info = model_cfg["regimes"][str(args.regime)]
    weight_class = regime_info["weight"]  # "heavy" or "light"
    d = model_cfg["defaults"]

    resolved = {
        "backbone": args.backbone,
        "regime": args.regime,
        "regime_weight_class": weight_class,
        "dataset": REGIME_DATASET[args.regime],
        "label_mode": args.label_mode,
        "mode": args.mode,
        "seed": args.seed if args.seed is not None else model_cfg["seeds"][weight_class][0],
        "epochs": args.epochs if args.epochs is not None else d[f"epochs_{weight_class}"],
        "batch_size": args.batch_size or d["batch_size"],
        "lr": args.lr if args.lr is not None else d["lr"],
        "weight_decay": args.weight_decay if args.weight_decay is not None else d["weight_decay"],
        "optimizer": args.optimizer or d["optimizer"],
        "lr_schedule": args.lr_schedule or d["lr_schedule"],
        "warmup_epochs": args.warmup_epochs if args.warmup_epochs is not None else d["warmup_epochs"],
        "class_weight_mode": args.class_weight_mode or d["class_weight_mode"],
        "norm_stats": args.norm_stats or d["norm_stats"],
        "pretrained_source": args.pretrained_source or regime_info["default_pretrained_source"] if args.pretrained_source is None else args.pretrained_source,
        "val_metric": args.val_metric or d["val_metric"],
        "patience": args.patience if args.patience is not None else d["patience"],
        "num_workers": args.num_workers if args.num_workers is not None else d["num_workers"],
        "amp": args.amp if args.amp is not None else d["amp"],
        "image_size": args.image_size or model_cfg["backbones"][args.backbone]["image_size"],
        "dropout": args.dropout if args.dropout is not None else d["dropout"],
        "loss_weight_crop": args.loss_weight_crop if args.loss_weight_crop is not None else d["loss_weight_crop"],
        "loss_weight_disease": args.loss_weight_disease if args.loss_weight_disease is not None else d["loss_weight_disease"],
        "manifests_dir": args.manifests_dir,
        "configs_dir": args.configs_dir,
        "runs_dir": args.runs_dir,
        "device": args.device,
        "dry_run": args.dry_run,
    }
    # args.pretrained_source default is None -> use regime's policy default ("auto" resolves
    # further, at model-build time, to imagenet/pvd_regime1 depending on regime).
    if args.pretrained_source is None:
        resolved["pretrained_source"] = "auto"
    return resolved


def get_device(choice: str):
    import torch
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def get_norm_stats(norm_stats_choice: str, dataset: str, manifests_dir: Path) -> tuple[tuple, tuple]:
    if norm_stats_choice == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    target = dataset if norm_stats_choice == "auto" else norm_stats_choice
    stats_path = manifests_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"--norm_stats={norm_stats_choice!r} needs {stats_path} (run scripts/run_data_pipeline.py first)."
        )
    stats = read_json(stats_path)
    cms = stats[target].get("channel_mean_std")
    if not cms:
        raise FileNotFoundError(
            f"dataset_stats.json has no channel_mean_std for {target!r} -- re-run the data "
            f"pipeline without --skip_channel_stats."
        )
    return tuple(cms["mean_rgb"]), tuple(cms["std_rgb"])


def build_dataloaders(cfg: dict, taxonomy, manifests_dir: Path):
    from torch.utils.data import DataLoader
    from src.data.dataset import PlantLeafDataset

    dataset_name = cfg["dataset"]
    manifest_path = manifests_dir / f"{dataset_name}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found -- run scripts/run_data_pipeline.py first.")

    mean, std = get_norm_stats(cfg["norm_stats"], dataset_name, manifests_dir)
    cfg["resolved_norm_mean"], cfg["resolved_norm_std"] = mean, std

    import pandas as pd
    full_df = pd.read_csv(manifest_path)
    train_df = full_df.loc[full_df["split"] == "train"].reset_index(drop=True)

    train_tf = build_transforms("train", image_size=cfg["image_size"], mean=mean, std=std)
    eval_tf = build_transforms("eval", image_size=cfg["image_size"], mean=mean, std=std)

    train_ds = PlantLeafDataset(full_df, split="train", label_mode=cfg["label_mode"], taxonomy=taxonomy, transform=train_tf)
    val_ds = PlantLeafDataset(full_df, split="val", label_mode=cfg["label_mode"], taxonomy=taxonomy, transform=eval_tf)
    test_ds = PlantLeafDataset(full_df, split="test", label_mode=cfg["label_mode"], taxonomy=taxonomy, transform=eval_tf)

    nw = cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)

    cfg["resolved_train_weights_source"] = train_df  # kept in-process only, never serialized
    return train_loader, val_loader, test_loader, train_df


def build_model(cfg: dict, taxonomy, device, logger: RunLogger):
    import torch
    from src.models.multitask_model import MultiHeadClassifier

    pretrained_source, ckpt_path = resolve_pretrained_source(
        cfg["pretrained_source"], cfg["regime"], cfg["runs_dir"], cfg["backbone"], cfg["label_mode"]
    )
    cfg["resolved_pretrained_source"] = pretrained_source
    cfg["resolved_pretrained_checkpoint"] = str(ckpt_path) if ckpt_path else None
    logger.log("pretrained_source_resolved", pretrained_source=pretrained_source, checkpoint=str(ckpt_path) if ckpt_path else None)

    model = MultiHeadClassifier(
        cfg["backbone"], num_crops=taxonomy.num_crops(), num_diseases=taxonomy.num_diseases(),
        pretrained=(pretrained_source == "imagenet"), dropout=cfg["dropout"],
    )
    if pretrained_source == "pvd_regime1":
        load_backbone_weights_only(ckpt_path, model)
        logger.log("regime1_backbone_loaded", checkpoint=str(ckpt_path))

    return model.to(device)


def build_optimizer_and_schedule(cfg: dict, model):
    import torch

    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=0.9)

    scheduler = None
    if cfg["lr_schedule"] == "cosine":
        warmup = cfg["warmup_epochs"]
        total = cfg["epochs"]

        def lr_lambda(epoch):
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / warmup
            import math
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def is_better(metric_name: str, value: float, best: float | None) -> bool:
    if best is None:
        return True
    if metric_name == "loss":
        return value < best
    return value > best


def train_and_select_best(cfg: dict, model, train_loader, val_loader, train_df, taxonomy, device, exp_dir: Path, logger: RunLogger):
    import csv
    import torch
    import torch.nn as nn

    crop_weights = compute_class_weights(train_df, "crop", taxonomy.crop_to_id, cfg["class_weight_mode"])
    disease_weights = compute_class_weights(train_df, "disease", taxonomy.disease_to_id, cfg["class_weight_mode"])
    cfg["resolved_crop_class_weights"] = crop_weights
    cfg["resolved_disease_class_weights"] = disease_weights

    crop_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(crop_weights, dtype=torch.float32, device=device))
    disease_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(disease_weights, dtype=torch.float32, device=device))

    optimizer, scheduler = build_optimizer_and_schedule(cfg, model)
    # torch.amp.GradScaler (not the deprecated torch.cuda.amp.GradScaler) still needs an
    # explicit device string even when disabled -- "cuda" here is just that label, not a
    # claim that CUDA is in use; enabled=False (set below whenever device.type != "cuda")
    # means it never actually touches the device.
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["amp"] and device.type == "cuda"))

    ckpt_dir = ensure_dir(exp_dir / "checkpoints")
    metrics_csv_path = exp_dir / "epoch_metrics.csv"
    csv_fields = [
        "epoch", "train_loss", "train_crop_loss", "train_disease_loss", "train_crop_acc", "train_disease_acc", "train_mean_acc",
        "val_loss", "val_crop_loss", "val_disease_loss", "val_crop_acc", "val_disease_acc", "val_mean_acc",
        "lr", "epoch_time_sec", "is_best",
    ]
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_fields)

    best_val_metric = None
    best_epoch = None
    epochs_without_improvement = 0
    n_epochs = 1 if cfg["dry_run"] else cfg["epochs"]

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loader_this_epoch = _maybe_limit(train_loader, cfg["dry_run"])
        val_loader_this_epoch = _maybe_limit(val_loader, cfg["dry_run"])

        train_result = run_epoch(model, train_loader_this_epoch, device, crop_loss_fn, disease_loss_fn,
                                  cfg["loss_weight_crop"], cfg["loss_weight_disease"],
                                  optimizer=optimizer, scaler=scaler, amp=cfg["amp"])
        val_result = run_epoch(model, val_loader_this_epoch, device, crop_loss_fn, disease_loss_fn,
                                cfg["loss_weight_crop"], cfg["loss_weight_disease"])
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.time() - t0

        current_metric = val_result[cfg["val_metric"]]
        improved = is_better(cfg["val_metric"], current_metric, best_val_metric)
        if improved:
            best_val_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch=epoch, extra={"val_metric": current_metric})
        else:
            epochs_without_improvement += 1
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch=epoch)

        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_result["loss"], train_result["crop_loss"], train_result["disease_loss"],
                train_result["crop_acc"], train_result["disease_acc"], train_result["mean_acc"],
                val_result["loss"], val_result["crop_loss"], val_result["disease_loss"],
                val_result["crop_acc"], val_result["disease_acc"], val_result["mean_acc"],
                optimizer.param_groups[0]["lr"], round(epoch_time, 2), improved,
            ])

        logger.log("epoch_end", epoch=epoch, train_loss=round(train_result["loss"], 4),
                   val_loss=round(val_result["loss"], 4), val_mean_acc=round(val_result["mean_acc"], 4),
                   is_best=improved, epoch_time_sec=round(epoch_time, 1))

        if cfg["patience"] > 0 and epochs_without_improvement >= cfg["patience"]:
            logger.log("early_stopped", epoch=epoch, epochs_without_improvement=epochs_without_improvement)
            break

    return best_epoch, best_val_metric


def _maybe_limit(loader, dry_run: bool):
    if not dry_run:
        return loader
    from itertools import islice
    return list(islice(loader, 2))  # 2 batches only


def run_test(model, test_loader, taxonomy, cfg, device):
    test_result = run_epoch(model, test_loader, device,
                             crop_loss_fn=__import__("torch").nn.CrossEntropyLoss(),
                             disease_loss_fn=__import__("torch").nn.CrossEntropyLoss(),
                             loss_weight_crop=cfg["loss_weight_crop"], loss_weight_disease=cfg["loss_weight_disease"],
                             collect_predictions=True)
    return build_test_report(test_result, taxonomy)


def main():
    args = parse_args()
    model_cfg = read_json(Path(args.configs_dir) / "model_config.json")
    cfg = resolve_config(args, model_cfg)

    exp_dir = ensure_dir(experiment_dir(cfg["runs_dir"], cfg["regime"], cfg["backbone"], cfg["label_mode"], cfg["seed"]))
    logger = RunLogger(exp_dir, "train")
    set_global_seed(cfg["seed"])
    device = get_device(cfg["device"])
    logger.log("start", **{k: v for k, v in cfg.items() if k != "resolved_train_weights_source"}, resolved_device=str(device))

    taxonomy = load_taxonomy(cfg["configs_dir"])
    manifests_dir = Path(cfg["manifests_dir"])

    if cfg["mode"] == "train":
        train_loader, val_loader, test_loader, train_df = build_dataloaders(cfg, taxonomy, manifests_dir)
        model = build_model(cfg, taxonomy, device, logger)
        best_epoch, best_val_metric = train_and_select_best(cfg, model, train_loader, val_loader, train_df, taxonomy, device, exp_dir, logger)

        write_json(exp_dir / "config.json", {k: v for k, v in cfg.items() if k != "resolved_train_weights_source"})

        if not cfg["dry_run"]:
            load_checkpoint(exp_dir / "checkpoints" / "best.pt", model)
            logger.log("testing_best_epoch", best_epoch=best_epoch, best_val_metric=best_val_metric)
            test_report = run_test(model, test_loader, taxonomy, cfg, device)
            write_json(exp_dir / "test_metrics.json", test_report)
            logger.log("test_complete",
                       crop_top1=test_report["crop"]["top1_accuracy"], crop_top3=test_report["crop"]["top3_accuracy"],
                       crop_balanced_acc=test_report["crop"]["balanced_accuracy"], crop_macro_f1=test_report["crop"]["macro_f1"],
                       disease_top1=test_report["disease"]["top1_accuracy"], disease_top3=test_report["disease"]["top3_accuracy"],
                       disease_balanced_acc=test_report["disease"]["balanced_accuracy"], disease_macro_f1=test_report["disease"]["macro_f1"])

        summary = {
            "backbone": cfg["backbone"], "regime": cfg["regime"], "seed": cfg["seed"], "label_mode": cfg["label_mode"],
            "best_epoch": best_epoch, "best_val_metric": best_val_metric, "val_metric_name": cfg["val_metric"],
            "dry_run": cfg["dry_run"],
        }
        write_json(exp_dir / "training_summary.json", summary)
        logger.log("done", **summary)

    else:  # mode == "test": standalone evaluation of an existing checkpoint
        ckpt_path = Path(args.checkpoint) if args.checkpoint else exp_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train this (backbone, regime, seed) first, or pass --checkpoint explicitly.")

        eval_dataset = args.eval_dataset or cfg["dataset"]
        is_cross_dataset = eval_dataset != cfg["dataset"]
        if args.eval_split is not None:
            eval_split = args.eval_split
        elif cfg["regime"] == 1 and eval_dataset == "plantdoc":
            # Regime 1 trained/validated on PlantVillage only -- no PlantDoc
            # split was ever touched, so none needs holding out here. Using
            # the complete dataset (not just its "test" slice) maximizes N
            # for this eval, which matters given PlantDoc's thin per-class
            # support (see assets/docs/05_modelling_decisions.md). Regime
            # 2/3 do NOT get this default: they trained/validated directly
            # on PlantDoc's train/val splits, so a standalone eval of THEM
            # still defaults to "test" to avoid leaking training data in.
            eval_split = "all"
        else:
            eval_split = "test"
        eval_manifest_path = manifests_dir / f"{eval_dataset}_manifest.csv"
        if not eval_manifest_path.exists():
            raise FileNotFoundError(f"{eval_manifest_path} not found -- run scripts/run_data_pipeline.py first.")

        import pandas as pd
        from torch.utils.data import DataLoader
        from src.data.dataset import PlantLeafDataset
        from src.models.multitask_model import MultiHeadClassifier

        mean, std = get_norm_stats(cfg["norm_stats"], eval_dataset, manifests_dir)
        eval_tf = build_transforms("eval", image_size=cfg["image_size"], mean=mean, std=std)
        eval_df = pd.read_csv(eval_manifest_path)
        eval_ds = PlantLeafDataset(eval_df, split=eval_split, label_mode=cfg["label_mode"], taxonomy=taxonomy, transform=eval_tf)
        eval_loader = DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

        model = MultiHeadClassifier(cfg["backbone"], num_crops=taxonomy.num_crops(), num_diseases=taxonomy.num_diseases(),
                                     pretrained=False, dropout=cfg["dropout"]).to(device)
        load_checkpoint(ckpt_path, model)

        logger.log("standalone_eval_start", checkpoint=str(ckpt_path), eval_dataset=eval_dataset, eval_split=eval_split,
                   is_cross_dataset=is_cross_dataset)
        test_report = run_test(model, eval_loader, taxonomy, cfg, device)
        out_name = f"eval_{eval_dataset}_{eval_split}.json"
        write_json(ckpt_path.parent.parent / out_name, test_report)
        logger.log("standalone_eval_complete",
                   crop_top1=test_report["crop"]["top1_accuracy"], crop_top3=test_report["crop"]["top3_accuracy"],
                   disease_top1=test_report["disease"]["top1_accuracy"], disease_top3=test_report["disease"]["top3_accuracy"],
                   output=str(ckpt_path.parent.parent / out_name))

    logger.close()


if __name__ == "__main__":
    main()
