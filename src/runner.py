"""
MIA Experiment Runner — multi-seed, multi-dataset.

CLI:
    python runner.py --datasets cifar10,cifar100 --seeds 42,14,12 --no-interactive

Full flow per (dataset, seed):
    1. Train plaintext target models
    2. Train shadow models
    3. Train basic attack models
    4. Fit LiRA distributions
    5. Convert plaintext → MPC (weight transfer)
    6. Evaluate plaintext models  (accuracy + basic MIA + LiRA + ROC)
    7. Evaluate MPC models        (accuracy + basic MIA + LiRA + ROC)

Then per dataset: aggregate across seeds, save JSON, generate charts.
"""

import argparse
import os
import copy
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from tqdm import tqdm

from config import ExperimentConfig, setup_dirs
from dataset_registry import get_dataset_spec, list_datasets, DatasetSpec
from models import PLAINTEXT_MODELS, MPC_MODELS
from training import plaintext_train_model
from conversion import convert_all_plaintext_to_mpc
from data import partition_dataset_for_mia, train_shadow_models, regenerate_shadow_indices
from attack import prepare_attack_dataset, train_attack_model
from lira import fit_lira_distributions
from evaluation import (
    evaluate_all_plaintext, evaluate_all_mpc,
    aggregate_across_seeds, print_results_summary, print_aggregated_summary,
    save_results, load_seed_results, save_seed_results,
)
from charts import generate_all_charts
from checkpointing import (
    check_plaintext_target_exists, check_shadow_models_exist,
    check_attack_model_exists, find_latest_intermediate_checkpoint,
    load_plaintext_model, load_shadow_models, load_attack_model,
    extract_epochs_from_checkpoint, check_all_training_complete,
    check_lira_params_exist, save_lira_params, load_lira_params,
)


# ═══════════════════════════════════════════════════════════════════════════
# Core: single (dataset, seed) experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_single_experiment(cfg: ExperimentConfig, ds: DatasetSpec,
                          seed: int, verbose: bool = True):
    """
    Run one full experiment for a single dataset and seed.

    Returns:
        results: {model_name: result_dict} (includes _raw_* keys for ROC)
    """
    DIRS = setup_dirs(dataset_name=ds.name, seed=seed)
    device = "cuda" if t.cuda.is_available() else "cpu"
    nc = ds.num_classes

    # ── data ──────────────────────────────────────────────────────────
    data_dir = os.path.join(DIRS["base"], "data")
    full_train, test_set = ds.load_fn(data_dir, ds.train_transform, ds.test_transform)

    # Eval copy (no augmentation) — reload with test transform
    full_eval, _ = ds.load_fn(data_dir, ds.test_transform, ds.test_transform)

    target_train_idx, target_test_idx, shadow_pool_idx = partition_dataset_for_mia(
        full_train, cfg.target_train_size, cfg.shadow_pool_ratio, seed)

    train_sub      = Subset(full_train, target_train_idx)
    train_sub_eval = Subset(full_eval,  target_train_idx)
    test_sub_eval  = Subset(full_eval,  target_test_idx)

    train_loader      = DataLoader(train_sub,      batch_size=cfg.batch_size,     shuffle=True,  num_workers=cfg.num_workers)
    test_loader       = DataLoader(test_set,       batch_size=cfg.batch_size,     shuffle=False, num_workers=cfg.num_workers)
    train_loader_eval = DataLoader(train_sub_eval, batch_size=cfg.batch_size,     shuffle=False, num_workers=cfg.num_workers)
    test_loader_eval  = DataLoader(test_sub_eval,  batch_size=cfg.batch_size,     shuffle=False, num_workers=cfg.num_workers)

    # MPC loaders
    train_mpc_eval = DataLoader(train_sub_eval, batch_size=cfg.mpc_batch_size, shuffle=False, num_workers=0)
    test_mpc_eval  = DataLoader(test_sub_eval,  batch_size=cfg.mpc_batch_size, shuffle=False, num_workers=0)
    test_mpc       = DataLoader(test_set,       batch_size=cfg.mpc_batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()

    plaintext_models   = {}
    shadow_all         = {}
    attack_models      = {}
    lira_params_all    = {}
    epochs_map         = {}

    # ── Check if all training is already complete ─────────────────────
    all_trained = check_all_training_complete(
        list(PLAINTEXT_MODELS.keys()), DIRS, cfg.num_shadow_models)
    if all_trained:
        print(f"\n[RESUME] All training artifacts found — fast-forwarding to evaluation.")

    # ── Phase 1: plaintext targets ────────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 1: PLAINTEXT TARGETS\n{'=' * 70}")

    for name, mcls in tqdm(PLAINTEXT_MODELS.items(), desc="Plaintext", disable=not verbose):
        print(f"\n[TARGET] {name}")
        n_ep = cfg.get_plaintext_epochs(name)
        exists, path = check_plaintext_target_exists(name, DIRS)

        if exists:
            print(f"  [SKIP] {path}")
            model = load_plaintext_model(mcls, path, device, nc)
            ep = extract_epochs_from_checkpoint(path) or n_ep
        else:
            rfound, rpath, repoch = find_latest_intermediate_checkpoint(name, DIRS["plaintext"])
            model = mcls(num_classes=nc).to(device)
            start, opt_state = 0, None
            if rfound:
                print(f"  [RESUME] epoch {repoch}")
                ckpt = t.load(rpath, map_location=device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    opt_state = ckpt.get("optimizer_state_dict")
                else:
                    model.load_state_dict(ckpt)
                start = repoch
            else:
                print(f"  [TRAIN] {n_ep} epochs")

            model, _, ep = plaintext_train_model(
                model, train_loader, n_ep, lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay, device=device,
                save_dir=DIRS["plaintext"], model_name=name, verbose=verbose,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                start_epoch=start, resume_optimizer_state=opt_state)

        plaintext_models[name] = model
        epochs_map[name] = ep

    # ── Phase 2: shadow models ────────────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 2: SHADOW MODELS\n{'=' * 70}")

    for name, mcls in tqdm(PLAINTEXT_MODELS.items(), desc="Shadows", disable=not verbose):
        print(f"\n[SHADOW] {name}")
        ep = epochs_map[name]
        exists, paths = check_shadow_models_exist(name, DIRS, cfg.num_shadow_models)

        if exists:
            print(f"  [SKIP] Loading {cfg.num_shadow_models}")
            shadows = load_shadow_models(mcls, paths, device, nc)
            indices = regenerate_shadow_indices(cfg.num_shadow_models, shadow_pool_idx, seed=seed)
        else:
            print(f"  [TRAIN] {cfg.num_shadow_models} × {ep} epochs")
            np.random.seed(seed)
            shadows, indices = train_shadow_models(
                cfg.num_shadow_models, mcls, full_train, shadow_pool_idx,
                name, DIRS["shadow_models"], ep, nc, device, verbose,
                lr=cfg.learning_rate)

        shadow_all[name] = (shadows, indices)

    # ── Phase 3: basic attack models ─────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 3: ATTACK MODELS\n{'=' * 70}")

    for name in tqdm(PLAINTEXT_MODELS, desc="Attacks", disable=not verbose):
        arch = name.replace("PlainText", "")
        print(f"\n[ATTACK] {arch}")
        exists, path = check_attack_model_exists(arch, DIRS)

        if exists:
            print(f"  [SKIP] {path}")
            atk = load_attack_model(path, device, nc)
        else:
            print(f"  [TRAIN] {cfg.attack_epochs} epochs")
            shadows, indices = shadow_all[name]
            X, y = prepare_attack_dataset(shadows, indices, full_eval, device, verbose)
            ap = os.path.join(DIRS["attack_models"], f"attack_{arch}.pt")
            atk = train_attack_model(X, y, nc, ap, cfg.attack_epochs, device, verbose)

        attack_models[arch] = atk

    # ── Phase 4: LiRA distributions ──────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 4: LiRA DISTRIBUTIONS\n{'=' * 70}")

    for name in tqdm(PLAINTEXT_MODELS, desc="LiRA", disable=not verbose):
        arch = name.replace("PlainText", "")
        print(f"\n[LiRA] {arch}")
        exists, lpath = check_lira_params_exist(arch, DIRS)

        if exists:
            print(f"  [SKIP] {lpath}")
            lira_params_all[arch] = load_lira_params(arch, DIRS)
        else:
            shadows, indices = shadow_all[name]
            lira_params_all[arch] = fit_lira_distributions(
                shadows, indices, full_eval, nc, device, verbose)
            save_lira_params(arch, lira_params_all[arch], DIRS)

    # ── Phase 5: convert → MPC ────────────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 5: PLAINTEXT → MPC CONVERSION\n{'=' * 70}")

    mpc_models = convert_all_plaintext_to_mpc(plaintext_models, nc, verbose)

    # ── Load cached eval results for resume ──────────────────────────
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(code_dir)
    results_dir = os.path.join(base_dir, "results", ds.name)
    cached = load_seed_results(results_dir, ds.name, seed)

    # Incremental save callback — merges new results into running total
    all_results = {}

    def _save_incremental(partial_results):
        all_results.update(partial_results)
        save_seed_results(all_results, results_dir, ds.name, seed)

    # ── Phase 6: evaluate plaintext ──────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 6: EVALUATE PLAINTEXT\n{'=' * 70}")

    pt_results = evaluate_all_plaintext(
        plaintext_models, attack_models, lira_params_all,
        test_loader, train_loader_eval, test_loader_eval,
        criterion, device, verbose,
        cached_results=cached, save_callback=_save_incremental)

    # ── Phase 7: evaluate MPC ────────────────────────────────────────
    print(f"\n{'=' * 70}\nPHASE 7: EVALUATE MPC\n{'=' * 70}")

    mpc_results = evaluate_all_mpc(
        mpc_models, attack_models, lira_params_all,
        test_mpc, train_mpc_eval, test_mpc_eval,
        criterion, "cpu", verbose,
        cached_results=cached, save_callback=_save_incremental)

    all_results = {**pt_results, **mpc_results}
    # Final incremental save with all results
    save_seed_results(all_results, results_dir, ds.name, seed)
    print_results_summary(all_results, f"SEED {seed} — {ds.name.upper()}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Outer loop: iterate datasets × seeds, aggregate, save, chart
# ═══════════════════════════════════════════════════════════════════════════

def run_full_experiment(cfg: ExperimentConfig, dataset_names: list[str],
                        seeds: list[int], verbose: bool = True):
    """
    Orchestrate multi-seed, multi-dataset experiments.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_dict = {
        "num_shadow_models": cfg.num_shadow_models,
        "target_train_size": cfg.target_train_size,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "attack_epochs": cfg.attack_epochs,
        "early_stopping_patience": cfg.early_stopping_patience,
        "early_stopping_min_delta": cfg.early_stopping_min_delta,
    }

    for ds_name in dataset_names:
        ds = get_dataset_spec(ds_name)
        print(f"\n{'#' * 70}")
        print(f"# DATASET: {ds.name.upper()}  |  {ds.num_classes} classes  |  "
              f"seeds: {seeds}")
        print(f"{'#' * 70}")

        per_seed = {}
        for seed in seeds:
            print(f"\n{'━' * 70}")
            print(f"  SEED {seed}  ×  {ds.name.upper()}")
            print(f"{'━' * 70}")

            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.seed = seed
            per_seed[seed] = run_single_experiment(cfg_copy, ds, seed, verbose)

        # Aggregate
        aggregated = aggregate_across_seeds(per_seed)
        print_aggregated_summary(aggregated, f"AGGREGATED — {ds.name.upper()}")

        # Determine output directory from DIRS structure
        code_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(code_dir)
        results_dir = os.path.join(base_dir, "results", ds.name)

        # Save JSON
        save_results(per_seed, aggregated, config_dict, seeds,
                     ds.name, results_dir, timestamp)

        # Charts
        print(f"\n[CHARTS] Generating for {ds.name}...")
        generate_all_charts(aggregated, per_seed, ds.name, results_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def interactive_config_loop(cfg):
    while True:
        cfg.display_table()
        print("\n  ENTER / 'y' → start  |  setting=val → modify  |  'q' → quit\n")
        try:
            inp = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return None
        if inp.lower() in ("q", "exit", "quit"):
            return None
        if inp in ("", "y"):
            return cfg
        for part in inp.replace(",", " ").split():
            if "=" in part:
                k, v = part.split("=", 1)
                ok, msg = cfg.set_field(k.strip(), v.strip())
                print(f"  {'[OK]' if ok else '[!]'} {msg}")


def parse_args():
    p = argparse.ArgumentParser(description="MIA Experiment — multi-seed, multi-dataset")

    # Dataset & seed control
    p.add_argument("--datasets", type=str, default="cifar10",
                   help=f"Comma-separated datasets ({', '.join(list_datasets())})")
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated random seeds (e.g., 42,14,12,2,0)")

    # Epoch budgets
    p.add_argument("--cnn-epochs",    type=int, default=200)
    p.add_argument("--mlp-epochs",    type=int, default=300)
    p.add_argument("--lenet-epochs",  type=int, default=150)
    p.add_argument("--mpc-cnn-epochs",   type=int, default=200)
    p.add_argument("--mpc-mlp-epochs",   type=int, default=300)
    p.add_argument("--mpc-lenet-epochs", type=int, default=150)

    # Shadow & attack
    p.add_argument("--attack-epochs",      type=int, default=60)
    p.add_argument("--num-shadow-models",  type=int, default=22)

    # Data
    p.add_argument("--target-train-size",  type=int, default=25000)
    p.add_argument("--shadow-pool-ratio",  type=float, default=0.7)

    # Training
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--mpc-batch-size", type=int,   default=32)
    p.add_argument("--learning-rate",  type=float, default=1e-2)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--num-workers",    type=int,   default=2)

    # Early stopping
    p.add_argument("--early-stopping-patience",  type=int,   default=30)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0005)

    # Flags
    p.add_argument("--quiet",          action="store_true")
    p.add_argument("--no-interactive", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    dataset_names = [s.strip() for s in args.datasets.split(",")]

    cfg = ExperimentConfig(
        cnn_epochs=args.cnn_epochs,
        mlp_epochs=args.mlp_epochs,
        lenet_epochs=args.lenet_epochs,
        mpc_cnn_epochs=args.mpc_cnn_epochs,
        mpc_mlp_epochs=args.mpc_mlp_epochs,
        mpc_lenet_epochs=args.mpc_lenet_epochs,
        attack_epochs=args.attack_epochs,
        num_shadow_models=args.num_shadow_models,
        target_train_size=args.target_train_size,
        batch_size=args.batch_size,
        mpc_batch_size=args.mpc_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        shadow_pool_ratio=args.shadow_pool_ratio,
        seed=seeds[0],
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    verbose = not args.quiet

    print(f"\nDatasets: {dataset_names}")
    print(f"Seeds:    {seeds}")

    if not args.no_interactive:
        cfg = interactive_config_loop(cfg)
        if cfg is None:
            return
    else:
        cfg.display_table()
        print("\nStarting experiment (--no-interactive)...")

    run_full_experiment(cfg, dataset_names, seeds, verbose)


if __name__ == "__main__":
    main()
