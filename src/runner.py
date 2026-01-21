import argparse
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import crypten
from tqdm import tqdm

from config import ExperimentConfig, setup_dirs
from models import PLAINTEXT_MODELS, MPC_MODELS
from training import plaintext_train_model, mpc_train_model
from data import partition_dataset_for_mia, train_shadow_models, regenerate_shadow_indices
from attack import (
    prepare_attack_dataset, 
    train_attack_model, 
    evaluate_mia_attack, 
    evaluate_accuracy_loss
)
from checkpointing import (
    check_plaintext_target_exists,
    check_mpc_target_exists,
    check_shadow_models_exist,
    check_attack_model_exists,
    load_plaintext_model,
    load_mpc_model,
    load_shadow_models,
    load_attack_model,
    get_attack_model_for_architecture,
    extract_epochs_from_checkpoint
)


def run_experiment(cfg: ExperimentConfig, verbose: bool = True):    
    DIRS = setup_dirs()
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # Standard CIFAR-10 values
    ])
    
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    target_train_idx, target_test_idx, shadow_pool_idx = partition_dataset_for_mia(
        full_dataset=full_dataset,
        target_train_size=cfg.target_train_size,
        shadow_pool_ratio=cfg.shadow_pool_ratio,
        seed=cfg.seed
    )

    target_train_subset = Subset(full_dataset, target_train_idx)
    target_test_subset = Subset(full_dataset, target_test_idx)

    target_train_loader = DataLoader(target_train_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    target_test_loader = DataLoader(target_test_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    target_train_loader_mpc = DataLoader(target_train_subset, batch_size=cfg.mpc_batch_size, shuffle=True, num_workers=cfg.num_workers)

    criterion = nn.CrossEntropyLoss()

    # Train plaintext targets, shadows, and attack models
    # With checkpoint-based resumption: skip already-trained models

    results = {}
    attack_models = {}

    # Plaintext Models
    plaintext_iter = tqdm(PLAINTEXT_MODELS.items(), desc="Plaintext Models", disable=not verbose)
    for name, model_class in plaintext_iter:
        plaintext_iter.set_postfix(model=name)
        print(f"\n{'='*60}")
        print(f"Processing model: {name}")
        print(f"{'='*60}")
        
        # Get architecture-specific epochs
        num_epochs = cfg.get_plaintext_epochs(name)
        arch_key = name.replace('PlainText', '')
        
        # Check if target model exists (look for complete checkpoint with _final.pt)
        target_exists, target_path = check_plaintext_target_exists(name, DIRS)
        
        if target_exists:
            print(f"[SKIP] Target model already trained, loading from: {target_path}")
            target_model = load_plaintext_model(model_class, target_path, device)
            # Extract epochs from checkpoint filename for shadow training
            epochs_trained = extract_epochs_from_checkpoint(target_path)
            if epochs_trained is None:
                epochs_trained = num_epochs  # Fallback to configured epochs
            print(f"Target was trained for {epochs_trained} epochs")
        else:
            print(f"[TRAIN] Training target model for {num_epochs} epochs...")
            model = model_class(num_classes=10).to(device)
            target_model, _, epochs_trained = plaintext_train_model(
                model, target_train_loader, num_epochs, 
                lr=cfg.learning_rate, device=device, 
                save_dir=DIRS['plaintext'], model_name=name,
                verbose=verbose,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta
            )
            print(f"Target model trained for {epochs_trained} epochs")
        
        test_loss, test_acc = evaluate_accuracy_loss(target_model, test_loader, criterion, device, verbose=verbose)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Check if shadow models exist
        shadows_exist, shadow_paths = check_shadow_models_exist(name, DIRS, cfg.num_shadow_models)
        
        if shadows_exist:
            print(f"[SKIP] Shadow models already trained, loading...")
            shadow_models = load_shadow_models(model_class, shadow_paths, device)
            # Regenerate indices with same seed for consistency
            shadow_indices = regenerate_shadow_indices(cfg.num_shadow_models, shadow_pool_idx, seed=cfg.seed)
        else:
            print(f"[TRAIN] Training {cfg.num_shadow_models} shadow models for {epochs_trained} epochs each...")
            # Reset the seed before training shadows to ensure reproducibility
            np.random.seed(cfg.seed)
            # Shadow models train for the same epochs as target (no early stopping)
            shadow_models, shadow_indices = train_shadow_models(
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
        
        # Check if attack model exists
        attack_exists, attack_path = check_attack_model_exists(arch_key, DIRS)
        
        if attack_exists:
            print(f"[SKIP] Attack model already trained, loading from: {attack_path}")
            attack_model = load_attack_model(attack_path, device)
        else:
            print(f"[TRAIN] Training attack model for {cfg.attack_epochs} epochs...")
            X_attack, y_attack = prepare_attack_dataset(shadow_models, shadow_indices, full_dataset, device=device, verbose=verbose)
            attack_path = f"{DIRS['attack_models']}/attack_{arch_key}.pt"
            attack_model = train_attack_model(
                X_attack, y_attack, 
                epochs=cfg.attack_epochs, 
                device=device, 
                save_path=attack_path,
                verbose=verbose
            )
        
        attack_models[arch_key] = attack_model
        
        print(f"[EVAL] Evaluating MIA attack...")
        acc, prec, rec = evaluate_mia_attack(
            target_model=target_model,
            attack_model=attack_model,
            train_loader=target_train_loader,
            test_loader=target_test_loader,
            device=device,
            is_mpc=False,
            verbose=verbose
        )
        results[name] = {
            'mia_accuracy': acc, 'mia_precision': prec, 'mia_recall': rec,
            'test_loss': test_loss, 'test_accuracy': test_acc, 'is_mpc': False
        }

    # MPC Models
    mpc_iter = tqdm(MPC_MODELS.items(), desc="MPC Models", disable=not verbose)
    for name, model_class in mpc_iter:
        mpc_iter.set_postfix(model=name)
        print(f"\n{'='*60}")
        print(f"Processing model: {name}")
        print(f"{'='*60}")
        
        # Get architecture-specific epochs
        num_epochs = cfg.get_mpc_epochs(name)
        
        # Check if MPC target model exists (look for complete checkpoint)
        mpc_exists, mpc_path = check_mpc_target_exists(name, DIRS)
        
        if mpc_exists:
            print(f"[SKIP] MPC target model already trained, loading from: {mpc_path}")
            target_model = load_mpc_model(model_class, mpc_path)
            epochs_trained = extract_epochs_from_checkpoint(mpc_path)
            if epochs_trained is None:
                epochs_trained = num_epochs
            print(f"MPC target was trained for {epochs_trained} epochs")
        else:
            print(f"[TRAIN] Training MPC target model for {num_epochs} epochs...")
            model = model_class(num_classes=10)
            target_model, _, epochs_trained = mpc_train_model(
                model, target_train_loader_mpc, num_epochs, lr=cfg.learning_rate,
                model_name=name, checkpoint_dir=DIRS['MPC'],
                verbose=verbose,
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta
            )
            print(f"MPC model trained for {epochs_trained} epochs")
        
        target_model.decrypt()
        
        # Get architecture key for attack model lookup
        # MpcCNN_Sigmoid -> CNN_Sigmoid
        arch_key = name.replace('Mpc', '')
        
        # Reuse attack model from plaintext model - NO training for MPC
        attack_model = get_attack_model_for_architecture(arch_key, attack_models, DIRS, device)
        
        if attack_model is not None:
            print(f"[REUSE] Using attack model from PlainText{arch_key}")
            
            print(f"[EVAL] Evaluating MIA attack (using decrypted model)...")
            # Use is_mpc=False since model is decrypted - runs fast plaintext inference
            acc, prec, rec = evaluate_mia_attack(
                target_model=target_model,
                attack_model=attack_model,
                train_loader=target_train_loader,  # Use regular loader, not MPC batch size
                test_loader=target_test_loader,
                device='cpu',  # Decrypted CrypTen models run on CPU
                is_mpc=False,   # Key change: use plaintext inference path
                verbose=verbose
            )
            
            # Evaluate test accuracy for MPC model
            test_loss, test_acc = evaluate_accuracy_loss(target_model, test_loader, criterion, 'cpu', verbose=verbose)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
            results[name] = {
                'mia_accuracy': acc, 'mia_precision': prec, 'mia_recall': rec,
                'test_loss': test_loss, 'test_accuracy': test_acc, 'is_mpc': True
            }
        else:
            print(f"[WARN] No attack model found for architecture {arch_key}")
            print(f"       Train PlainText{arch_key} first to generate the attack model.")
            
            # Still evaluate test accuracy
            test_loss, test_acc = evaluate_accuracy_loss(target_model, test_loader, criterion, 'cpu', verbose=verbose)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
            results[name] = {
                'mia_accuracy': None, 'mia_precision': None, 'mia_recall': None,
                'test_loss': test_loss, 'test_accuracy': test_acc, 'is_mpc': True
            }
    print_results_summary(results) 
    return results

def print_results_summary(results):
    print(f"\n{'='*90}")
    print("RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<25} {'Type':<10} {'Test Acc':<10} {'Test Loss':<10} {'MIA Acc':<10} {'MIA Prec':<10} {'MIA Rec':<10}")
    print("-" * 90)
    for name, res in sorted(results.items()):
        t_type = 'MPC' if res['is_mpc'] else 'Plaintext'
        mia_acc = f"{res['mia_accuracy']:.2f}" if res['mia_accuracy'] is not None else 'N/A'
        mia_prec = f"{res['mia_precision']:.2f}" if res['mia_precision'] is not None else 'N/A'
        mia_rec = f"{res['mia_recall']:.2f}" if res['mia_recall'] is not None else 'N/A'
        print(f"{name:<25} {t_type:<10} {res['test_accuracy']:<10.2f} {res['test_loss']:<10.4f} {mia_acc:<10} {mia_prec:<10} {mia_rec:<10}")


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
    
    # Split by comma or whitespace, but keep key=value together
    # Replace commas with spaces first
    input_str = input_str.replace(',', ' ')
    
    # Split by whitespace
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
    while True:
        # Display current configuration
        cfg.display_table()
        
        # Prompt user
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
        
        # Check for quit
        if user_input.lower() in ['q', 'exit', 'quit']:
            print("Exiting without training.")
            return None
        
        # Check for start training
        if user_input == '' or user_input.lower() == 'y':
            print("\nStarting training...")
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
    parser.add_argument('--cnn-epochs', type=int, default=80,
                        help='Number of epochs for CNN model training (default: 80)')
    parser.add_argument('--mlp-epochs', type=int, default=120,
                        help='Number of epochs for MLP model training (default: 120)')
    parser.add_argument('--lenet-epochs', type=int, default=40,
                        help='Number of epochs for LeNet model training (default: 40)')
    parser.add_argument('--mpc-cnn-epochs', type=int, default=80,
                        help='Number of epochs for MPC CNN model training (default: 80)')
    parser.add_argument('--mpc-mlp-epochs', type=int, default=120,
                        help='Number of epochs for MPC MLP model training (default: 120)')
    parser.add_argument('--mpc-lenet-epochs', type=int, default=40,
                        help='Number of epochs for MPC LeNet model training (default: 40)')
    parser.add_argument('--attack-epochs', type=int, default=30,
                        help='Number of epochs for attack model training (default: 30)')
    parser.add_argument('--num-shadow-models', type=int, default=9,
                        help='Number of shadow models to train (default: 9)')
    parser.add_argument('--target-train-size', type=int, default=10000,
                        help='Size of target model training set (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for plaintext training (default: 128)')
    parser.add_argument('--mpc-batch-size', type=int, default=32,
                        help='Batch size for MPC training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--shadow-pool-ratio', type=float, default=0.5,
                        help='Ratio of data for shadow pool (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loader workers (default: 2)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience for target models (default: 10)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                        help='Early stopping minimum delta for target models (default: 0.001)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Skip interactive configuration and start training immediately')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create initial config from command line args
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
        shadow_pool_ratio=args.shadow_pool_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    
    verbose = not args.quiet
    
    # Interactive configuration loop (unless --no-interactive)
    if not args.no_interactive:
        cfg = interactive_config_loop(cfg)
        if cfg is None:
            return  # User quit
    else:
        cfg.display_table()
        print("\nStarting training (--no-interactive mode)...")
    
    run_experiment(cfg, verbose=verbose)

if __name__ == '__main__':
    main()
