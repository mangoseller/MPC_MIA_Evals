"""
MIA Experiment Runner

Flow:
1. Train all plaintext target models
2. Train all shadow models
3. Train all attack models
4. Train all MPC target models
5. Evaluate all plaintext models
6. Evaluate all MPC models
7. Save results to JSON
"""

import argparse
import json
import os
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm

from config import ExperimentConfig, setup_dirs
from models import PLAINTEXT_MODELS, MPC_MODELS
from training import plaintext_train_model, mpc_train_model
from data import partition_dataset_for_mia, train_shadow_models, regenerate_shadow_indices
from attack import prepare_attack_dataset, train_attack_model
from evaluation import (
    evaluate_all_plaintext_models,
    evaluate_all_mpc_models,
    print_results_summary,
    save_results
)
from checkpointing import (
    check_plaintext_target_exists,
    check_mpc_target_exists,
    check_shadow_models_exist,
    check_attack_model_exists,
    find_latest_intermediate_checkpoint,
    load_plaintext_model,
    load_mpc_model,
    load_shadow_models,
    load_attack_model,
    extract_epochs_from_checkpoint
)


def run_experiment(cfg: ExperimentConfig, verbose: bool = True):
    """
    Run the full MIA experiment with the flow:
    1. Train all plaintext target models
    2. Train all shadow models
    3. Train all attack models
    4. Train all MPC target models
    5. Evaluate all plaintext models
    6. Evaluate all MPC models
    7. Save results to JSON
    """
    DIRS = setup_dirs()
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    full_dataset_eval = datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform)
    
    target_train_idx, target_test_idx, shadow_pool_idx = partition_dataset_for_mia(
        full_dataset=full_dataset,
        target_train_size=cfg.target_train_size,
        shadow_pool_ratio=cfg.shadow_pool_ratio,
        seed=cfg.seed
    )

    # Training subsets (with augmentation)
    target_train_subset = Subset(full_dataset, target_train_idx)
    
    # Evaluation subsets (no augmentation) for consistent MIA evaluation
    target_train_subset_eval = Subset(full_dataset_eval, target_train_idx)
    target_test_subset_eval = Subset(full_dataset_eval, target_test_idx)

    # Data loaders
    target_train_loader = DataLoader(target_train_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    target_train_loader_eval = DataLoader(target_train_subset_eval, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    target_test_loader_eval = DataLoader(target_test_subset_eval, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # MPC-specific loaders (smaller batch size)
    target_train_loader_mpc = DataLoader(target_train_subset, batch_size=cfg.mpc_batch_size, shuffle=True, num_workers=cfg.num_workers)
    target_train_loader_mpc_eval = DataLoader(target_train_subset_eval, batch_size=cfg.mpc_batch_size, shuffle=False, num_workers=0)
    target_test_loader_mpc_eval = DataLoader(target_test_subset_eval, batch_size=cfg.mpc_batch_size, shuffle=False, num_workers=0)
    test_loader_mpc = DataLoader(test_dataset, batch_size=cfg.mpc_batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    criterion = nn.CrossEntropyLoss()

    # Storage for trained models
    plaintext_models = {}
    mpc_models = {}
    shadow_models_all = {}
    attack_models = {}
    epochs_trained_map = {}


    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING PLAINTEXT TARGET MODELS")
    print("=" * 70)
    
    plaintext_iter = tqdm(PLAINTEXT_MODELS.items(), desc="Plaintext Targets", disable=not verbose)
    for name, model_class in plaintext_iter:
        plaintext_iter.set_postfix(model=name)
        print(f"\n[TARGET] {name}")
        
        num_epochs = cfg.get_plaintext_epochs(name)
        target_exists, target_path = check_plaintext_target_exists(name, DIRS)

        if target_exists:
            print(f"  [SKIP] Loading from: {target_path}")
            model = load_plaintext_model(model_class, target_path, device)
            epochs_trained = extract_epochs_from_checkpoint(target_path) or num_epochs
        else:
            # Check for intermediate checkpoint to resume from
            resume_found, resume_path, resume_epoch = find_latest_intermediate_checkpoint(name, DIRS['plaintext'])

            model = model_class(num_classes=10).to(device)
            start_epoch = 0
            resume_optimizer_state = None

            if resume_found:
                print(f"  [RESUME] Resuming from epoch {resume_epoch}: {resume_path}")
                checkpoint_data = t.load(resume_path, map_location=device)
                if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                    model.load_state_dict(checkpoint_data['model_state_dict'])
                    resume_optimizer_state = checkpoint_data.get('optimizer_state_dict')
                else:
                    # Old format: raw state_dict
                    model.load_state_dict(checkpoint_data)
                start_epoch = resume_epoch
            else:
                print(f"  [TRAIN] Training for {num_epochs} epochs...")

            model, _, epochs_trained = plaintext_train_model(
                model, target_train_loader, num_epochs,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                device=device,
                save_dir=DIRS['plaintext'],
                model_name=name,
                verbose=verbose,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                start_epoch=start_epoch,
                resume_optimizer_state=resume_optimizer_state
            )
        
        plaintext_models[name] = model
        epochs_trained_map[name] = epochs_trained
        print(f"  Epochs trained: {epochs_trained}")


    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING SHADOW MODELS")
    print("=" * 70)
    
    shadow_iter = tqdm(PLAINTEXT_MODELS.items(), desc="Shadow Models", disable=not verbose)
    for name, model_class in shadow_iter:
        shadow_iter.set_postfix(model=name)
        print(f"\n[SHADOW] {name}")
        
        epochs_trained = epochs_trained_map[name]
        shadows_exist, shadow_paths = check_shadow_models_exist(name, DIRS, cfg.num_shadow_models)
        
        if shadows_exist:
            print(f"  [SKIP] Loading {cfg.num_shadow_models} shadow models...")
            shadows = load_shadow_models(model_class, shadow_paths, device)
            indices = regenerate_shadow_indices(cfg.num_shadow_models, shadow_pool_idx, seed=cfg.seed)
        else:
            print(f"  [TRAIN] Training {cfg.num_shadow_models} shadow models for {epochs_trained} epochs each...")
            np.random.seed(cfg.seed)
            shadows, indices = train_shadow_models(
                num_shadows=cfg.num_shadow_models,
                model_class=model_class,
                full_dataset=full_dataset,
                shadow_pool_indices=shadow_pool_idx,
                model_name=name,
                base_dir=DIRS['shadow_models'],
                num_epochs=epochs_trained,
                device=device,
                verbose=verbose
            )
        
        shadow_models_all[name] = (shadows, indices)


    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING ATTACK MODELS")
    print("=" * 70)
    
    attack_iter = tqdm(PLAINTEXT_MODELS.items(), desc="Attack Models", disable=not verbose)
    for name, model_class in attack_iter:
        arch_key = name.replace('PlainText', '')
        attack_iter.set_postfix(model=arch_key)
        print(f"\n[ATTACK] {arch_key}")
        
        attack_exists, attack_path = check_attack_model_exists(arch_key, DIRS)
        
        if attack_exists:
            print(f"  [SKIP] Loading from: {attack_path}")
            attack_model = load_attack_model(attack_path, device)
        else:
            print(f"  [TRAIN] Training for {cfg.attack_epochs} epochs...")
            shadows, indices = shadow_models_all[name]
            X_attack, y_attack = prepare_attack_dataset(
                shadows, indices, full_dataset_eval, device=device, verbose=verbose
            )
            attack_path = f"{DIRS['attack_models']}/attack_{arch_key}.pt"
            attack_model = train_attack_model(
                X_attack, y_attack,
                epochs=cfg.attack_epochs,
                device=device,
                save_path=attack_path,
                verbose=verbose
            )
        
        attack_models[arch_key] = attack_model


    print("\n" + "=" * 70)
    print("PHASE 4: TRAINING MPC TARGET MODELS")
    print("=" * 70)
    
    mpc_iter = tqdm(MPC_MODELS.items(), desc="MPC Targets", disable=not verbose)
    for name, model_class in mpc_iter:
        mpc_iter.set_postfix(model=name)
        print(f"\n[MPC TARGET] {name}")
        
        num_epochs = cfg.get_mpc_epochs(name)
        mpc_exists, mpc_path = check_mpc_target_exists(name, DIRS)

        if mpc_exists:
            print(f"  [SKIP] Loading from: {mpc_path}")
            model = load_mpc_model(model_class, mpc_path)
            epochs_trained = extract_epochs_from_checkpoint(mpc_path) or num_epochs
        else:
            # Check for intermediate checkpoint to resume from
            resume_found, resume_path, resume_epoch = find_latest_intermediate_checkpoint(name, DIRS['MPC'])

            model = model_class(num_classes=10)
            start_epoch = 0
            mpc_resume_path = None

            if resume_found:
                print(f"  [RESUME] Resuming from epoch {resume_epoch}: {resume_path}")
                start_epoch = resume_epoch
                mpc_resume_path = resume_path
            else:
                print(f"  [TRAIN] Training for {num_epochs} epochs...")

            model, _, epochs_trained = mpc_train_model(
                model, target_train_loader_mpc, num_epochs,
                lr=cfg.learning_rate,
                model_name=name,
                checkpoint_dir=DIRS['MPC'],
                verbose=verbose,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
                start_epoch=start_epoch,
                resume_path=mpc_resume_path
            )
        
        # Ensure model is decrypted for later evaluation
        if hasattr(model, 'decrypt'):
            model.decrypt()
        
        mpc_models[name] = model
        print(f"  Epochs trained: {epochs_trained}")


    print("\n" + "=" * 70)
    print("PHASE 5: EVALUATING PLAINTEXT MODELS")
    print("=" * 70)
    
    plaintext_results = evaluate_all_plaintext_models(
        models=plaintext_models,
        attack_models=attack_models,
        test_loader=test_loader,
        train_loader_eval=target_train_loader_eval,
        test_loader_eval=target_test_loader_eval,
        criterion=criterion,
        device=device,
        verbose=verbose
    )


    print("\n" + "=" * 70)
    print("PHASE 6: EVALUATING MPC MODELS")
    print("=" * 70)
    
    mpc_results = evaluate_all_mpc_models(
        models=mpc_models,
        attack_models=attack_models,
        test_loader_mpc=test_loader_mpc,
        train_loader_mpc_eval=target_train_loader_mpc_eval,
        test_loader_mpc_eval=target_test_loader_mpc_eval,
        criterion=criterion,
        device='cpu',
        verbose=verbose
    )


    all_results = {**plaintext_results, **mpc_results}
    
    print_results_summary(all_results)
    
    # Save to JSON with config metadata
    results_dir = os.path.join(os.path.dirname(DIRS['plaintext']), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_shadow_models': cfg.num_shadow_models,
            'target_train_size': cfg.target_train_size,
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'weight_decay': cfg.weight_decay,
            'seed': cfg.seed,
        },
        'results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


def parse_setting_updates(input_str: str) -> list[tuple[str, str]]:
    """
    Parse setting updates from user input.
    
    Supports formats like:
        mlp_epochs=100
        learning_rate=0.002, seed=12, num_workers=4
        learning_rate=0.002 seed=12 num_workers=4
    
    Returns:
        List of (key, value) tuples
    """
    updates = []
    input_str = input_str.replace(',', ' ')
    parts = input_str.split()
    
    for part in parts:
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                updates.append((key, value))
    
    return updates


def interactive_config_loop(cfg: ExperimentConfig) -> ExperimentConfig:
    """Interactive loop for configuring experiment parameters."""
    while True:
        cfg.display_table()
        
        print("\nOptions:")
        print("  - Press ENTER or type 'y' to start training with these settings")
        print("  - Type setting=value to modify (e.g., mlp_epochs=100)")
        print("  - Multiple settings: learning_rate=0.002, seed=12, num_workers=4")
        print("  - Type 'q' to quit without training")
        print()
        
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return None
        
        if user_input.lower() in ['q', 'exit', 'quit']:
            print("Exiting without training.")
            return None
        
        if user_input == '' or user_input.lower() == 'y':
            print("\nStarting experiment...")
            return cfg
        
        updates = parse_setting_updates(user_input)
        
        if not updates:
            print("\n[!] Invalid input format. Use setting=value (e.g., mlp_epochs=100)")
            continue
        
        for key, value in updates:
            success, message = cfg.set_field(key, value)
            if success:
                print(f"  [OK] {message}")
            else:
                print(f"  [!] {message}")


def parse_args():
    parser = argparse.ArgumentParser(description='MIA Experiment on Plaintext and MPC Models')
    parser.add_argument('--cnn-epochs', type=int, default=100,
                        help='Number of epochs for CNN model training (default: 100)')
    parser.add_argument('--mlp-epochs', type=int, default=150,
                        help='Number of epochs for MLP model training (default: 150)')
    parser.add_argument('--lenet-epochs', type=int, default=60,
                        help='Number of epochs for LeNet model training (default: 60)')
    parser.add_argument('--mpc-cnn-epochs', type=int, default=100,
                        help='Number of epochs for MPC CNN model training (default: 100)')
    parser.add_argument('--mpc-mlp-epochs', type=int, default=150,
                        help='Number of epochs for MPC MLP model training (default: 150)')
    parser.add_argument('--mpc-lenet-epochs', type=int, default=60,
                        help='Number of epochs for MPC LeNet model training (default: 60)')
    parser.add_argument('--attack-epochs', type=int, default=30,
                        help='Number of epochs for attack model training (default: 30)')
    parser.add_argument('--num-shadow-models', type=int, default=16,
                        help='Number of shadow models to train (default: 16)')
    parser.add_argument('--target-train-size', type=int, default=20000,
                        help='Size of target model training set (default: 20000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for plaintext training (default: 128)')
    parser.add_argument('--mpc-batch-size', type=int, default=32,
                        help='Batch size for MPC training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--shadow-pool-ratio', type=float, default=0.5,
                        help='Ratio of data for shadow pool (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loader workers (default: 2)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Early stopping patience for target models (default: 15)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0005,
                        help='Early stopping minimum delta for target models (default: 0.0005)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Skip interactive configuration and start training immediately')
    return parser.parse_args()


def main():
    args = parse_args()
    
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
        seed=args.seed,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    verbose = not args.quiet
    
    if not args.no_interactive:
        cfg = interactive_config_loop(cfg)
        if cfg is None:
            return
    else:
        cfg.display_table()
        print("\nStarting experiment (--no-interactive mode)...")
    
    run_experiment(cfg, verbose=verbose)


if __name__ == '__main__':
    main()
