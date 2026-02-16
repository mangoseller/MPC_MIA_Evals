"""
Likelihood Ratio Attack (LiRA) — offline variant.

Carlini et al., "Membership Inference Attacks From First Principles" (S&P 2022).

Offline variant: fit per-class Gaussian reference distributions from shadow
models, then score target samples via a likelihood-ratio test.
"""

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import crypten
from tqdm import tqdm
import gc


# ── helpers ───────────────────────────────────────────────────────────────

def _logit_transform(p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _gaussian_log_prob(x, mu, sigma):
    return -0.5 * np.log(2.0 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2


# ── Phase 1: fit distributions ───────────────────────────────────────────

def fit_lira_distributions(
    shadow_models, shadow_indices, full_dataset,
    num_classes=10, device="cuda", verbose=True,
):
    """
    Build per-class Gaussian reference distributions for LiRA.

    Returns:
        lira_params: {class_id: {in_mean, in_std, out_mean, out_std}}
    """
    class_in = {c: [] for c in range(num_classes)}
    class_out = {c: [] for c in range(num_classes)}

    with t.no_grad():
        for i, model in tqdm(enumerate(shadow_models), total=len(shadow_models),
                             desc="Fitting LiRA", disable=not verbose):
            model.to(device).eval()
            train_idx, test_idx = shadow_indices[i]
            train_set = set(train_idx.tolist() if hasattr(train_idx, "tolist") else train_idx)

            all_idx = np.concatenate([train_idx, test_idx])
            loader = DataLoader(Subset(full_dataset, all_idx), batch_size=128,
                                shuffle=False, num_workers=0)

            all_probs, all_labels = [], []
            for inputs, labels in loader:
                probs = F.softmax(model(inputs.to(device)), dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels)

            all_probs = t.cat(all_probs)
            all_labels = t.cat(all_labels)

            for j, idx in enumerate(all_idx):
                c = all_labels[j].item()
                logit_conf = _logit_transform(all_probs[j][c].item())
                (class_in if idx in train_set else class_out)[c].append(logit_conf)

            model.cpu()
            t.cuda.empty_cache()

    params = {}
    for c in range(num_classes):
        in_arr, out_arr = np.array(class_in[c]), np.array(class_out[c])
        params[c] = {
            "in_mean":  float(in_arr.mean()),  "in_std":  float(max(in_arr.std(), 1e-6)),
            "out_mean": float(out_arr.mean()), "out_std": float(max(out_arr.std(), 1e-6)),
        }

    if verbose:
        print("\nLiRA per-class distributions:")
        for c in range(num_classes):
            p = params[c]
            print(f"  Class {c:>3d}:  IN N({p['in_mean']:+.3f}, {p['in_std']:.3f})  "
                  f"OUT N({p['out_mean']:+.3f}, {p['out_std']:.3f})  "
                  f"Δμ={p['in_mean'] - p['out_mean']:.3f}")

    return params


# ── Phase 2: evaluate ────────────────────────────────────────────────────

def evaluate_lira(
    target_model, lira_params, train_loader, test_loader,
    device, is_mpc=False, verbose=True,
):
    """
    Evaluate LiRA membership inference attack.

    Returns:
        dict with accuracy, precision, recall, and raw scores/labels for ROC.
    """
    if not is_mpc:
        target_model.eval()

    def _collect(loader, is_member, desc=""):
        scores, classes, membership = [], [], []

        if is_mpc and hasattr(target_model, "encrypted") and not target_model.encrypted:
            target_model.encrypt()

        for inputs, labels in tqdm(loader, desc=desc, leave=False, disable=not verbose):
            if is_mpc:
                with t.no_grad():
                    x = crypten.cryptensor(inputs)
                    out = target_model(x)
                    probs = F.softmax(out.get_plain_text(), dim=1)
                    del x, out
                gc.collect()
            else:
                inputs = inputs.to(device)
                with t.no_grad():
                    probs = F.softmax(target_model(inputs), dim=1)

            probs_np = probs.cpu().numpy()
            labels_np = labels.numpy()
            for k in range(len(labels_np)):
                c = int(labels_np[k])
                scores.append(_logit_transform(probs_np[k][c]))
                classes.append(c)
                membership.append(1.0 if is_member else 0.0)

        return scores, classes, membership

    m_s, m_c, m_l = _collect(train_loader, True, "LiRA Members")
    nm_s, nm_c, nm_l = _collect(test_loader, False, "LiRA Non-members")

    all_scores = np.array(m_s + nm_s)
    all_classes = m_c + nm_c
    all_labels = np.array(m_l + nm_l)

    # Compute log-likelihood ratios
    log_lr = np.zeros(len(all_scores))
    for i in range(len(all_scores)):
        p = lira_params[all_classes[i]]
        log_lr[i] = (_gaussian_log_prob(all_scores[i], p["in_mean"], p["in_std"])
                    - _gaussian_log_prob(all_scores[i], p["out_mean"], p["out_std"]))

    preds = (log_lr > 0.0).astype(float)
    accuracy = 100.0 * (preds == all_labels).sum() / len(all_labels)

    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": precision,
        "recall": recall,
        "raw_scores": log_lr,       # for ROC computation
        "raw_labels": all_labels,    # for ROC computation
    }
