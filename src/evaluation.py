"""
Unified evaluation — accuracy + basic MIA + LiRA, with ROC metrics.

Every evaluation function returns raw scores alongside summary metrics so
the chart module can render ROC curves across seeds.
"""

import json
import os
from datetime import datetime
from typing import Optional
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import crypten
from tqdm import tqdm
import gc

from sklearn.metrics import roc_curve, roc_auc_score

from lira import evaluate_lira


# ═══════════════════════════════════════════════════════════════════════════
# ROC utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_roc_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute ROC curve and TPR at fixed low-FPR thresholds."""
    if len(np.unique(labels)) < 2:
        return {"auc": 0.5, "tpr_at_01pct_fpr": 0.0, "tpr_at_1pct_fpr": 0.0,
                "tpr_at_10pct_fpr": 0.0, "fpr": [0, 1], "tpr": [0, 1]}

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    return {
        "auc": auc,
        "tpr_at_01pct_fpr": float(np.interp(0.001, fpr, tpr)),
        "tpr_at_1pct_fpr":  float(np.interp(0.01,  fpr, tpr)),
        "tpr_at_10pct_fpr": float(np.interp(0.10,  fpr, tpr)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_accuracy(model, loader, criterion, device, verbose=True):
    model.eval()
    loss_sum = correct = total = 0
    with t.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False, disable=not verbose):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


def evaluate_accuracy_mpc(model, loader, criterion, verbose=True):
    if not crypten.is_initialized():
        crypten.init()

    loss_sum = correct = total = 0
    was_enc = getattr(model, "encrypted", False)
    if not was_enc:
        model.encrypt()

    try:
        with t.no_grad():
            for x, y in tqdm(loader, desc="Eval MPC", leave=False, disable=not verbose):
                x = x.cpu() if x.is_cuda else x
                x_enc = crypten.cryptensor(x)
                out = model(x_enc).get_plain_text()
                loss_sum += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                del x_enc, out
                gc.collect()
    finally:
        if not was_enc:
            model.decrypt()

    return loss_sum / len(loader), 100.0 * correct / total


# ═══════════════════════════════════════════════════════════════════════════
# Basic MIA  (returns raw scores for ROC)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_mia(target_model, attack_model, train_loader, test_loader,
                 device, is_mpc=False, verbose=True) -> dict:
    """
    Returns dict with accuracy/precision/recall/ROC metrics AND raw_scores/raw_labels.
    """
    if is_mpc and not crypten.is_initialized():
        crypten.init()

    if not is_mpc:
        target_model.eval()

    atk_orig_dev = next(attack_model.parameters()).device
    attack_model = attack_model.to(device)
    attack_model.eval()

    def _preds(loader, is_member, desc=""):
        preds, labels = [], []
        if is_mpc and hasattr(target_model, "encrypted") and not target_model.encrypted:
            target_model.encrypt()
        for inputs, _ in tqdm(loader, desc=desc, leave=False, disable=not verbose):
            if is_mpc:
                inputs = inputs.cpu() if inputs.is_cuda else inputs
                with t.no_grad():
                    xe = crypten.cryptensor(inputs)
                    out = target_model(xe)
                    bp = F.softmax(out.get_plain_text(), dim=1)
                    del xe, out
                gc.collect()
            else:
                inputs = inputs.to(device)
                with t.no_grad():
                    bp = F.softmax(target_model(inputs), dim=1)
            preds.append(bp.cpu())
            labels.extend([1.0 if is_member else 0.0] * inputs.size(0))
        return t.cat(preds), t.tensor(labels).unsqueeze(1)

    mp, ml = _preds(train_loader, True, "Members")
    nmp, nml = _preds(test_loader, False, "Non-members")
    all_preds = t.cat([mp, nmp])
    all_labels = t.cat([ml, nml])

    with t.no_grad():
        attack_probs = attack_model(all_preds.to(device)).cpu().numpy().squeeze()

    labels_np = all_labels.numpy().squeeze()

    binary_preds = (attack_probs > 0.5).astype(float)
    accuracy = 100.0 * (binary_preds == labels_np).sum() / len(labels_np)
    tp = ((binary_preds == 1) & (labels_np == 1)).sum()
    fp = ((binary_preds == 1) & (labels_np == 0)).sum()
    fn = ((binary_preds == 0) & (labels_np == 1)).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    roc = compute_roc_metrics(attack_probs, labels_np)

    attack_model.to(atk_orig_dev)

    return {
        "accuracy": float(accuracy),
        "precision": precision,
        "recall": recall,
        **{f"roc_{k}": v for k, v in roc.items()},      # roc_auc, roc_tpr_at_1pct_fpr, ...
        "raw_scores": attack_probs,   # ndarray – stripped before JSON
        "raw_labels": labels_np,      # ndarray – stripped before JSON
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single-model evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_plaintext(
    name, model, attack_model, lira_params,
    test_loader, train_loader_eval, test_loader_eval,
    criterion, device, verbose=True,
) -> dict:
    print(f"\n[EVAL] {name}")

    test_loss, test_acc = evaluate_accuracy(model, test_loader, criterion, device, verbose)
    train_loss, train_acc = evaluate_accuracy(model, train_loader_eval, criterion, device, verbose)
    overfit = train_acc - test_acc
    print(f"  Test {test_acc:.2f}% | Train {train_acc:.2f}% | Gap {overfit:+.2f}%")

    basic = evaluate_mia(model, attack_model, train_loader_eval, test_loader_eval,
                         device, is_mpc=False, verbose=verbose)
    print(f"  Basic MIA  Acc {basic['accuracy']:.2f}% AUC {basic['roc_auc']:.4f} "
          f"TPR@1%FPR {basic['roc_tpr_at_1pct_fpr']:.4f}")

    lira = evaluate_lira(model, lira_params, train_loader_eval, test_loader_eval,
                         device, is_mpc=False, verbose=verbose)
    lira_roc = compute_roc_metrics(lira["raw_scores"], lira["raw_labels"])
    print(f"  LiRA       Acc {lira['accuracy']:.2f}% AUC {lira_roc['auc']:.4f} "
          f"TPR@1%FPR {lira_roc['tpr_at_1pct_fpr']:.4f}")

    return _build_result(name, False,
                         test_loss, test_acc, train_loss, train_acc, overfit,
                         basic, lira, lira_roc)


def evaluate_single_mpc(
    name, model, attack_model, lira_params,
    test_loader_mpc, train_loader_mpc, test_loader_mpc_eval,
    criterion, device, verbose=True,
) -> dict:
    print(f"\n[EVAL] {name}")

    try:
        test_loss, test_acc = evaluate_accuracy_mpc(model, test_loader_mpc, criterion, verbose)
        print(f"  Test {test_acc:.2f}%")

        basic = evaluate_mia(model, attack_model, train_loader_mpc, test_loader_mpc_eval,
                             device="cpu", is_mpc=True, verbose=verbose)
        print(f"  Basic MIA  Acc {basic['accuracy']:.2f}% AUC {basic['roc_auc']:.4f} "
              f"TPR@1%FPR {basic['roc_tpr_at_1pct_fpr']:.4f}")

        lira = evaluate_lira(model, lira_params, train_loader_mpc, test_loader_mpc_eval,
                             device="cpu", is_mpc=True, verbose=verbose)
        lira_roc = compute_roc_metrics(lira["raw_scores"], lira["raw_labels"])
        print(f"  LiRA       Acc {lira['accuracy']:.2f}% AUC {lira_roc['auc']:.4f} "
              f"TPR@1%FPR {lira_roc['tpr_at_1pct_fpr']:.4f}")
    finally:
        # Always ensure model is decrypted so subsequent evals aren't corrupted
        if hasattr(model, "encrypted") and model.encrypted:
            model.decrypt()
        gc.collect()

    return _build_result(name, True,
                         test_loss, test_acc, None, None, None,
                         basic, lira, lira_roc)


def _build_result(name, is_mpc, test_loss, test_acc, train_loss, train_acc, overfit,
                  basic, lira, lira_roc):
    """Assemble the result dict from sub-dicts."""
    return {
        "model_name": name, "is_mpc": is_mpc,
        "test_loss": test_loss, "test_accuracy": test_acc,
        "train_loss": train_loss, "train_accuracy": train_acc, "overfit_gap": overfit,
        # Basic MIA
        "basic_mia_accuracy":   basic["accuracy"],
        "basic_mia_precision":  basic["precision"],
        "basic_mia_recall":     basic["recall"],
        "basic_mia_auc":        basic["roc_auc"],
        "basic_tpr_at_01pct_fpr": basic["roc_tpr_at_01pct_fpr"],
        "basic_tpr_at_1pct_fpr":  basic["roc_tpr_at_1pct_fpr"],
        "basic_tpr_at_10pct_fpr": basic["roc_tpr_at_10pct_fpr"],
        # LiRA
        "lira_accuracy":   lira["accuracy"],
        "lira_precision":  lira["precision"],
        "lira_recall":     lira["recall"],
        "lira_auc":        lira_roc["auc"],
        "lira_tpr_at_01pct_fpr": lira_roc["tpr_at_01pct_fpr"],
        "lira_tpr_at_1pct_fpr":  lira_roc["tpr_at_1pct_fpr"],
        "lira_tpr_at_10pct_fpr": lira_roc["tpr_at_10pct_fpr"],
        # Raw scores for ROC curves (stripped before JSON serialization)
        "_basic_raw_scores": basic["raw_scores"],
        "_basic_raw_labels": basic["raw_labels"],
        "_lira_raw_scores":  lira["raw_scores"],
        "_lira_raw_labels":  lira["raw_labels"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Batch evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_all_plaintext(
    models, attack_models, lira_params_all,
    test_loader, train_eval, test_eval,
    criterion, device, verbose=True,
    cached_results=None, save_callback=None,
):
    print("\n" + "=" * 60 + "\nEVALUATING PLAINTEXT MODELS\n" + "=" * 60)
    results = {}
    cached_results = cached_results or {}
    for name, model in tqdm(models.items(), desc="Plaintext", disable=not verbose):
        if name in cached_results:
            print(f"\n[EVAL] {name}  [SKIP — cached]")
            results[name] = cached_results[name]
            continue
        arch = name.replace("PlainText", "")
        atk = attack_models.get(arch)
        lp = lira_params_all.get(arch)
        if atk is None:
            print(f"[WARN] No attack model for {name}, skip")
            continue
        results[name] = evaluate_single_plaintext(
            name, model, atk, lp, test_loader, train_eval, test_eval,
            criterion, device, verbose)
        if save_callback is not None:
            save_callback(results)
    return results


def evaluate_all_mpc(
    models, attack_models, lira_params_all,
    test_mpc, train_mpc, test_mpc_eval,
    criterion, device, verbose=True,
    cached_results=None, save_callback=None,
):
    print("\n" + "=" * 60 + "\nEVALUATING MPC MODELS\n" + "=" * 60)
    results = {}
    cached_results = cached_results or {}
    for name, model in tqdm(models.items(), desc="MPC", disable=not verbose):
        if name in cached_results:
            print(f"\n[EVAL] {name}  [SKIP — cached]")
            results[name] = cached_results[name]
            continue
        arch = name.replace("Mpc", "")
        atk = attack_models.get(arch)
        lp = lira_params_all.get(arch)
        if hasattr(model, "decrypt"):
            model.decrypt()
        if atk is None:
            print(f"[WARN] No attack model for {name}, skip")
            continue
        try:
            results[name] = evaluate_single_mpc(
                name, model, atk, lp, test_mpc, train_mpc, test_mpc_eval,
                criterion, device, verbose)
        except Exception as e:
            print(f"\n[ERROR] MPC eval failed for {name}: {e}")
            print(f"  Skipping {name} and continuing with remaining models.")
            if hasattr(model, "decrypt"):
                try:
                    model.decrypt()
                except Exception:
                    pass
            gc.collect()
            continue
        if save_callback is not None:
            save_callback(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation across seeds
# ═══════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = [
    "test_accuracy", "train_accuracy", "overfit_gap",
    "basic_mia_accuracy", "basic_mia_precision", "basic_mia_recall",
    "basic_mia_auc", "basic_tpr_at_01pct_fpr", "basic_tpr_at_1pct_fpr", "basic_tpr_at_10pct_fpr",
    "lira_accuracy", "lira_precision", "lira_recall",
    "lira_auc", "lira_tpr_at_01pct_fpr", "lira_tpr_at_1pct_fpr", "lira_tpr_at_10pct_fpr",
]


def aggregate_across_seeds(per_seed: dict[int, dict]) -> dict:
    """
    per_seed: {seed: {model_name: result_dict}}
    Returns:  {model_name: {metric_mean, metric_std, ...}}
    """
    all_names = set()
    for sr in per_seed.values():
        all_names.update(sr.keys())

    agg = {}
    for name in sorted(all_names):
        entry = {"model_name": name, "is_mpc": None, "n_seeds": 0}
        for key in _METRIC_KEYS:
            vals = []
            for seed, sr in per_seed.items():
                if name in sr and sr[name].get(key) is not None:
                    vals.append(sr[name][key])
                    if entry["is_mpc"] is None:
                        entry["is_mpc"] = sr[name].get("is_mpc", False)
            entry["n_seeds"] = max(entry["n_seeds"], len(vals))
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            entry[f"{key}_std"]  = float(np.std(vals))  if vals else None
        agg[name] = entry
    return agg


def strip_raw_scores(results: dict) -> dict:
    """Remove numpy arrays before JSON serialization."""
    clean = {}
    for name, r in results.items():
        clean[name] = {k: v for k, v in r.items() if not k.startswith("_")}
    return clean


# ═══════════════════════════════════════════════════════════════════════════
# Results I/O & display
# ═══════════════════════════════════════════════════════════════════════════

def print_results_summary(results: dict, title: str = "RESULTS"):
    """Print table for a single seed's results."""
    hdr = (f"{'Model':<25} {'Type':<6} {'Test%':<7} {'Overfit':<8} "
           f"{'B-Acc%':<7} {'B-AUC':<7} {'B-TPR1%':<8} "
           f"{'L-Acc%':<7} {'L-AUC':<7} {'L-TPR1%':<8}")
    sep = "=" * len(hdr)
    print(f"\n{sep}\n  {title}\n{sep}\n{hdr}\n{'-' * len(hdr)}")

    def _f(v, fmt=".2f"):
        return f"{v:{fmt}}" if v is not None else "—"

    for n in sorted(results):
        r = results[n]
        tp = "MPC" if r.get("is_mpc") else "PT"
        print(f"{n:<25} {tp:<6} {_f(r.get('test_accuracy')):<7} {_f(r.get('overfit_gap')):<8} "
              f"{_f(r.get('basic_mia_accuracy')):<7} {_f(r.get('basic_mia_auc'), '.4f'):<7} "
              f"{_f(r.get('basic_tpr_at_1pct_fpr'), '.4f'):<8} "
              f"{_f(r.get('lira_accuracy')):<7} {_f(r.get('lira_auc'), '.4f'):<7} "
              f"{_f(r.get('lira_tpr_at_1pct_fpr'), '.4f'):<8}")
    print(sep)


def print_aggregated_summary(agg: dict, title: str = "AGGREGATED RESULTS"):
    """Print table with mean ± std across seeds."""
    hdr = (f"{'Model':<25} {'Type':<6} {'Test% (±)':<12} "
           f"{'B-AUC (±)':<12} {'B-TPR1% (±)':<14} "
           f"{'L-AUC (±)':<12} {'L-TPR1% (±)':<14}")
    sep = "=" * len(hdr)
    print(f"\n{sep}\n  {title}\n{sep}\n{hdr}\n{'-' * len(hdr)}")

    def _ms(name, key):
        m = agg[name].get(f"{key}_mean")
        s = agg[name].get(f"{key}_std")
        if m is None:
            return "—"
        if s is not None and s > 0:
            return f"{m:.2f}±{s:.2f}"
        return f"{m:.2f}"

    def _ms4(name, key):
        m = agg[name].get(f"{key}_mean")
        s = agg[name].get(f"{key}_std")
        if m is None:
            return "—"
        if s is not None and s > 0:
            return f"{m:.4f}±{s:.4f}"
        return f"{m:.4f}"

    for n in sorted(agg):
        tp = "MPC" if agg[n].get("is_mpc") else "PT"
        print(f"{n:<25} {tp:<6} {_ms(n, 'test_accuracy'):<12} "
              f"{_ms4(n, 'basic_mia_auc'):<12} {_ms4(n, 'basic_tpr_at_1pct_fpr'):<14} "
              f"{_ms4(n, 'lira_auc'):<12} {_ms4(n, 'lira_tpr_at_1pct_fpr'):<14}")
    print(sep)


def _seed_results_path(results_dir: str, dataset_name: str, seed: int) -> str:
    """Stable path for per-seed incremental results (not timestamped)."""
    return os.path.join(results_dir, f"seed_{seed}_results.json")


def load_seed_results(results_dir: str, dataset_name: str, seed: int) -> dict:
    """
    Load previously saved per-seed evaluation results.

    Returns:
        dict mapping model_name → result_dict (without raw arrays),
        or empty dict if no cached results found.
    """
    path = _seed_results_path(results_dir, dataset_name, seed)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cached = data.get("results", {})
        print(f"  [CACHE] Loaded {len(cached)} cached eval results from {path}")
        return cached
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [WARN] Failed to load cached results: {e}")
        return {}


def save_seed_results(results: dict, results_dir: str, dataset_name: str, seed: int):
    """
    Save per-seed evaluation results incrementally to a stable (non-timestamped) JSON.
    Strips raw numpy arrays before writing.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = _seed_results_path(results_dir, dataset_name, seed)

    clean = {}
    for name, r in results.items():
        clean[name] = {k: v for k, v in r.items() if not k.startswith("_")}

    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "seed": seed,
        "results": clean,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def save_results(per_seed, aggregated, config_dict, seeds, dataset_name, output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"results_{timestamp}.json")

    # Strip raw numpy arrays from per-seed data
    clean_per_seed = {str(s): strip_raw_scores(r) for s, r in per_seed.items()}

    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "seeds": seeds,
        "config": config_dict,
        "aggregated": aggregated,
        "per_seed": clean_per_seed,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {filepath}")
    return filepath
