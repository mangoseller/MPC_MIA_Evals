"""
MIA experiment charts.

Generates:
  1. Bar charts — MIA accuracy per architecture (plaintext vs MPC)
  2. Bar charts — TPR @ 1% FPR per architecture
  3. ROC curves — mean ± std across architectures, plaintext vs MPC
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── visual constants ──────────────────────────────────────────────────────
_PT_C  = "#3A76AF"   # steel blue  — plaintext
_MPC_C = "#E8793A"   # warm coral  — MPC
_BASE  = 50.0

_ARCH_ORDER = [
    "CNN_Sigmoid",  "CNN_Tanh",  "CNN_GELU",
    "MLP_Sigmoid",  "MLP_Tanh",  "MLP_GELU",
    "LeNet_Sigmoid","LeNet_Tanh","LeNet_GELU",
]
_FAM_BOUNDS = [3, 6]


def _arch(name):
    return name.replace("PlainText", "").replace("Mpc", "")


# ═══════════════════════════════════════════════════════════════════════════
# Grouped bar chart  (supports error bars from aggregated data)
# ═══════════════════════════════════════════════════════════════════════════

def _paired(results, mean_key, std_key=None):
    """Extract paired (pt, mpc) values and optional error bars."""
    pt, mpc, pt_err, mpc_err = {}, {}, {}, {}
    for name, r in results.items():
        arch = _arch(name)
        val = r.get(mean_key)
        err = r.get(std_key) if std_key else None
        if val is None:
            continue
        if r.get("is_mpc"):
            mpc[arch] = val
            if err is not None:
                mpc_err[arch] = err
        else:
            pt[arch] = val
            if err is not None:
                pt_err[arch] = err

    keys = [a for a in _ARCH_ORDER if a in pt or a in mpc]
    for a in sorted((set(pt) | set(mpc)) - set(keys)):
        keys.append(a)

    pv  = [pt.get(a, float("nan"))  for a in keys]
    mv  = [mpc.get(a, float("nan")) for a in keys]
    pe  = [pt_err.get(a, 0)  for a in keys] if pt_err  else None
    me  = [mpc_err.get(a, 0) for a in keys] if mpc_err else None
    return keys, pv, mv, pe, me


def _bar_chart(keys, pv, mv, pe, me, title, ylabel, path,
               show_baseline=True, y_bottom=None):
    n = len(keys)
    if n == 0:
        return
    x = np.arange(n)
    w = 0.34
    g = 0.04

    fig, ax = plt.subplots(figsize=(max(10, n * 1.15), 5.4))

    bar_kw = dict(edgecolor="white", linewidth=0.6)
    ax.bar(x - w/2 - g, pv, w, yerr=pe, capsize=3, label="Plaintext",
           color=_PT_C, error_kw=dict(lw=1), **bar_kw)
    ax.bar(x + w/2 + g, mv, w, yerr=me, capsize=3, label="MPC",
           color=_MPC_C, error_kw=dict(lw=1), **bar_kw)

    if show_baseline:
        ax.axhline(_BASE, color="#888", lw=0.9, ls="--", zorder=1)
        ax.text(n - 0.5, _BASE + 0.6, "random chance", ha="right", va="bottom",
                fontsize=8, color="#888")

    for b in _FAM_BOUNDS:
        if b < n:
            ax.axvline(b - 0.5, color="#ccc", lw=0.7, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=12)

    vals = [v for v in pv + mv if not np.isnan(v)]
    if y_bottom is not None:
        lo = y_bottom
    else:
        lo = max(0, min(vals, default=50) - 6)
    ax.set_ylim(lo, max(vals, default=55) + 6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", lw=0.5, color="#e0e0e0")
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#ccc", fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ROC curve  (mean ± std band, plaintext vs MPC)
# ═══════════════════════════════════════════════════════════════════════════

def _interpolate_roc(per_seed_results, model_names, score_key, label_key):
    """
    Interpolate TPR to a common FPR grid for a set of models across seeds.
    Returns (mean_tpr, std_tpr) at each point on the grid.
    """
    from sklearn.metrics import roc_curve as sk_roc

    grid = np.logspace(-4, 0, 500)
    tpr_list = []

    for seed_results in per_seed_results.values():
        for mn in model_names:
            if mn not in seed_results:
                continue
            r = seed_results[mn]
            scores = r.get(score_key)
            labels = r.get(label_key)
            if scores is None or labels is None:
                continue
            fpr, tpr, _ = sk_roc(labels, scores)
            tpr_interp = np.interp(grid, fpr, tpr)
            tpr_list.append(tpr_interp)

    if not tpr_list:
        return grid, np.zeros_like(grid), np.zeros_like(grid)

    tpr_arr = np.array(tpr_list)
    return grid, tpr_arr.mean(axis=0), tpr_arr.std(axis=0)


def _roc_chart(per_seed_results, pt_names, mpc_names, score_key, label_key,
               title, path):
    """ROC with mean ± std bands for plaintext vs MPC."""
    grid_pt,  mean_pt,  std_pt  = _interpolate_roc(per_seed_results, pt_names,  score_key, label_key)
    grid_mpc, mean_mpc, std_mpc = _interpolate_roc(per_seed_results, mpc_names, score_key, label_key)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.plot(grid_pt,  mean_pt,  color=_PT_C,  lw=2,   label="Plaintext (mean)")
    ax.fill_between(grid_pt,  mean_pt - std_pt,  mean_pt + std_pt,
                    color=_PT_C,  alpha=0.15)
    ax.plot(grid_mpc, mean_mpc, color=_MPC_C, lw=2,   label="MPC (mean)")
    ax.fill_between(grid_mpc, mean_mpc - std_mpc, mean_mpc + std_mpc,
                    color=_MPC_C, alpha=0.15)

    ax.plot([1e-4, 1], [1e-4, 1], ls="--", color="#aaa", lw=0.8, label="Random")

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=10)
    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.9, edgecolor="#ccc")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", lw=0.4, color="#e0e0e0")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_charts(aggregated, per_seed_results, dataset_name, output_dir):
    """
    Generate all charts for one dataset.

    Args:
        aggregated:        {model_name: {metric_mean, metric_std, ...}}
        per_seed_results:  {seed: {model_name: result_dict_with_raw_scores}}
        dataset_name:      e.g. "cifar10"
        output_dir:        directory for chart PNGs
    """
    os.makedirs(output_dir, exist_ok=True)
    ds = dataset_name.upper()
    saved = []

    # ── bar charts (from aggregated data) ─────────────────────────────

    def _bar(metric_root, chart_title, ylabel, fname, baseline=True, y_bot=None):
        k, pv, mv, pe, me = _paired(aggregated, f"{metric_root}_mean", f"{metric_root}_std")
        if k:
            p = os.path.join(output_dir, fname)
            _bar_chart(k, pv, mv, pe, me, f"{ds} — {chart_title}", ylabel, p,
                       show_baseline=baseline, y_bottom=y_bot)
            saved.append(p)

    _bar("basic_mia_accuracy", "Shadow-Model MIA Accuracy", "MIA Accuracy (%)",
         "mia_basic_accuracy.png")
    _bar("lira_accuracy", "LiRA Accuracy", "MIA Accuracy (%)",
         "mia_lira_accuracy.png")
    _bar("basic_tpr_at_1pct_fpr", "Shadow-Model MIA — TPR @ 1% FPR",
         "TPR @ 1% FPR", "tpr_basic_1pct.png", baseline=False, y_bot=0)
    _bar("lira_tpr_at_1pct_fpr", "LiRA — TPR @ 1% FPR",
         "TPR @ 1% FPR", "tpr_lira_1pct.png", baseline=False, y_bot=0)
    _bar("basic_mia_auc", "Shadow-Model MIA — AUC", "AUC",
         "auc_basic.png", baseline=False, y_bot=0.4)
    _bar("lira_auc", "LiRA — AUC", "AUC",
         "auc_lira.png", baseline=False, y_bot=0.4)

    # ── ROC curves (from per-seed raw scores) ─────────────────────────

    all_names = set()
    for sr in per_seed_results.values():
        all_names.update(sr.keys())

    pt_names  = sorted(n for n in all_names if not aggregated.get(n, {}).get("is_mpc", True))
    mpc_names = sorted(n for n in all_names if aggregated.get(n, {}).get("is_mpc", False))

    if pt_names or mpc_names:
        p = os.path.join(output_dir, "roc_basic.png")
        _roc_chart(per_seed_results, pt_names, mpc_names,
                   "_basic_raw_scores", "_basic_raw_labels",
                   f"{ds} — Shadow-Model MIA ROC", p)
        saved.append(p)

        p = os.path.join(output_dir, "roc_lira.png")
        _roc_chart(per_seed_results, pt_names, mpc_names,
                   "_lira_raw_scores", "_lira_raw_labels",
                   f"{ds} — LiRA ROC", p)
        saved.append(p)

    return saved
