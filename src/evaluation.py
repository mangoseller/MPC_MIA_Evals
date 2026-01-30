import json
import os
from datetime import datetime
from typing import Optional
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import crypten
from tqdm import tqdm
import gc

def evaluate_accuracy(model, data_loader, criterion, device, verbose=True) -> tuple[float, float]:

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loader_iter = tqdm(data_loader, desc="Evaluating", leave=False, disable=not verbose)
    with t.no_grad():
        for inputs, targets in loader_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate_accuracy_mpc(model, data_loader, criterion, verbose=True) -> tuple[float, float]:

    running_loss = 0.0
    correct = 0
    total = 0

    was_encrypted = getattr(model, 'encrypted', False)
    if not was_encrypted:
        model.encrypt()

    loader_iter = tqdm(data_loader, desc="Evaluating MPC", leave=False, disable=not verbose)
    for inputs, targets in loader_iter:
        x_enc = crypten.cryptensor(inputs)
        
        with t.no_grad():
            output_enc = model(x_enc)
            outputs = output_enc.get_plain_text()
        
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    if not was_encrypted:
        model.decrypt()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate_mia(
    target_model, 
    attack_model, 
    train_loader, 
    test_loader, 
    device, 
    is_mpc=False, 
    verbose=True
) -> tuple[float, float, float]:
    """
    Evaluate MIA attack effectiveness.
    
    Args:
        target_model: The model being attacked
        attack_model: The trained attack model
        train_loader: DataLoader for member samples (training data)
        test_loader: DataLoader for non-member samples (test data)
        device: Device for computation
        is_mpc: Whether target_model is an MPC model
        verbose: Show progress bars
    
    Returns:
        (accuracy, precision, recall)
    """
    if not is_mpc:
        target_model.eval()
    attack_model.eval()
    
    def get_predictions(loader, is_member, desc=""):
        preds = []
        labels = []

        was_decrypted = False
        if is_mpc and hasattr(target_model, 'encrypted') and not target_model.encrypted:
            target_model.encrypt()
            was_decrypted = True
        
        loader_iter = tqdm(loader, desc=desc, leave=False, disable=not verbose)
        for inputs, _ in loader_iter:
            if is_mpc:
                with t.no_grad():
                    x_input = crypten.cryptensor(inputs)
                    output_enc = target_model(x_input)
                    batch_preds = F.softmax(output_enc.get_plain_text(), dim=1)
                    del x_input, output_enc
                gc.collect()
            else:
                inputs = inputs.to(device)
                with t.no_grad():
                    outputs = target_model(inputs)
                    batch_preds = F.softmax(outputs, dim=1)
            
            preds.append(batch_preds.cpu())
            labels.extend([1.0 if is_member else 0.0] * inputs.size(0))
            
        return t.cat(preds), t.tensor(labels).unsqueeze(1)
    
    member_preds, member_labels = get_predictions(train_loader, is_member=True, desc="Members")
    non_member_preds, non_member_labels = get_predictions(test_loader, is_member=False, desc="Non-members")
    
    all_preds = t.cat([member_preds, non_member_preds])
    all_labels = t.cat([member_labels, non_member_labels])
    
    with t.no_grad():
        attack_probs = attack_model(all_preds.to(device))
        attack_preds = (attack_probs > 0.5).float().cpu()
        
    correct = (attack_preds == all_labels).sum().item()
    accuracy = 100.0 * correct / all_labels.size(0)
    
    true_positives = ((attack_preds == 1) & (all_labels == 1)).sum().item()
    false_positives = ((attack_preds == 1) & (all_labels == 0)).sum().item()
    false_negatives = ((attack_preds == 0) & (all_labels == 1)).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return accuracy, precision, recall


def evaluate_single_plaintext_model(
    name: str,
    model,
    attack_model,
    test_loader: DataLoader,
    train_loader_eval: DataLoader,
    test_loader_eval: DataLoader,
    criterion,
    device: str,
    verbose: bool = True
) -> dict:
    """
    Fully evaluate a single plaintext model.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"\n[EVAL] Evaluating {name}...")
    
    # Test accuracy
    test_loss, test_acc = evaluate_accuracy(model, test_loader, criterion, device, verbose)
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Training accuracy (for overfitting measurement)
    train_loss, train_acc = evaluate_accuracy(model, train_loader_eval, criterion, device, verbose)
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    overfit_gap = train_acc - test_acc
    print(f"  Overfitting gap: {overfit_gap:.2f}%")
    
    # MIA evaluation
    mia_acc, mia_prec, mia_rec = evaluate_mia(
        target_model=model,
        attack_model=attack_model,
        train_loader=train_loader_eval,
        test_loader=test_loader_eval,
        device=device,
        is_mpc=False,
        verbose=verbose
    )
    print(f"  MIA Acc: {mia_acc:.2f}% | Precision: {mia_prec:.4f} | Recall: {mia_rec:.4f}")
    
    return {
        'model_name': name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'overfit_gap': overfit_gap,
        'mia_accuracy': mia_acc,
        'mia_precision': mia_prec,
        'mia_recall': mia_rec,
        'is_mpc': False
    }


def evaluate_single_mpc_model(
    name: str,
    model,
    attack_model,
    test_loader_mpc: DataLoader,
    train_loader_mpc_eval: DataLoader,
    test_loader_mpc_eval: DataLoader,
    criterion,
    device: str,
    verbose: bool = True
) -> dict:
    """
    Fully evaluate a single MPC model.
    
    Returns:
        Dictionary with all evaluation metrics, or partial results if no attack model
    """
    print(f"\n[EVAL] Evaluating {name}...")
    
    # Test accuracy
    test_loss, test_acc = evaluate_accuracy_mpc(model, test_loader_mpc, criterion, verbose)
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    if attack_model is None:
        print(f"  [WARN] No attack model available, skipping MIA evaluation")
        return {
            'model_name': name,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'train_loss': None,
            'train_accuracy': None,
            'overfit_gap': None,
            'mia_accuracy': None,
            'mia_precision': None,
            'mia_recall': None,
            'is_mpc': True
        }
    
    # MIA evaluation
    mia_acc, mia_prec, mia_rec = evaluate_mia(
        target_model=model,
        attack_model=attack_model,
        train_loader=train_loader_mpc_eval,
        test_loader=test_loader_mpc_eval,
        device=device,  # MPC models use CPU after decryption
        is_mpc=True,
        verbose=verbose
    )
    print(f"  MIA Acc: {mia_acc:.2f}% | Precision: {mia_prec:.4f} | Recall: {mia_rec:.4f}")
    
    return {
        'model_name': name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'train_loss': None,
        'train_accuracy': None,
        'overfit_gap': None,
        'mia_accuracy': mia_acc,
        'mia_precision': mia_prec,
        'mia_recall': mia_rec,
        'is_mpc': True
    }


def evaluate_all_plaintext_models(
    models: dict,
    attack_models: dict,
    test_loader: DataLoader,
    train_loader_eval: DataLoader,
    test_loader_eval: DataLoader,
    criterion,
    device: str,
    verbose: bool = True
) -> dict:
    """
    Evaluate all plaintext models.
    
    Args:
        models: Dict of {name: model}
        attack_models: Dict of {arch_key: attack_model}
        test_loader: Test data loader
        train_loader_eval: Training data loader (non-augmented) for MIA
        test_loader_eval: Test data loader for MIA (from training split)
        criterion: Loss function
        device: Computation device
        verbose: Show progress
    
    Returns:
        Dict of {model_name: results_dict}
    """
    print("\n" + "=" * 60)
    print("EVALUATING PLAINTEXT MODELS")
    print("=" * 60)
    
    results = {}
    
    for name, model in tqdm(models.items(), desc="Plaintext Evaluation", disable=not verbose):
        arch_key = name.replace('PlainText', '')
        attack_model = attack_models.get(arch_key)
        
        if attack_model is None:
            print(f"[WARN] No attack model for {name}, skipping...")
            continue
        
        result = evaluate_single_plaintext_model(
            name=name,
            model=model,
            attack_model=attack_model,
            test_loader=test_loader,
            train_loader_eval=train_loader_eval,
            test_loader_eval=test_loader_eval,
            criterion=criterion,
            device=device,
            verbose=verbose
        )
        results[name] = result
    
    return results


def evaluate_all_mpc_models(
    models: dict,
    attack_models: dict,
    test_loader_mpc: DataLoader,
    train_loader_mpc_eval: DataLoader,
    test_loader_mpc_eval: DataLoader,
    criterion,
    device: str,
    verbose: bool = True
) -> dict:
    """
    Evaluate all MPC models.
    
    Args:
        models: Dict of {name: model}
        attack_models: Dict of {arch_key: attack_model}
        test_loader_mpc: Test data loader (MPC batch size)
        train_loader_mpc_eval: Training data loader for MIA (MPC batch size)
        test_loader_mpc_eval: Test data loader for MIA (MPC batch size)
        criterion: Loss function
        device: Computation device (typically 'cpu' for MPC)
        verbose: Show progress
    
    Returns:
        Dict of {model_name: results_dict}
    """
    print("\n" + "=" * 60)
    print("EVALUATING MPC MODELS")
    print("=" * 60)
    
    results = {}
    
    for name, model in tqdm(models.items(), desc="MPC Evaluation", disable=not verbose):
        # MpcCNN_Sigmoid -> CNN_Sigmoid
        arch_key = name.replace('Mpc', '')
        attack_model = attack_models.get(arch_key)
        
        # Ensure model is decrypted for evaluation
        if hasattr(model, 'decrypt'):
            model.decrypt()
        
        result = evaluate_single_mpc_model(
            name=name,
            model=model,
            attack_model=attack_model,
            test_loader_mpc=test_loader_mpc,
            train_loader_mpc_eval=train_loader_mpc_eval,
            test_loader_mpc_eval=test_loader_mpc_eval,
            criterion=criterion,
            device=device,
            verbose=verbose
        )
        results[name] = result
    
    return results


def print_results_summary(results: dict):
    """Print a formatted summary table of results."""
    print(f"\n{'=' * 120}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 120}")
    print(f"{'Model':<25} {'Type':<10} {'Test Acc':<10} {'Train Acc':<10} {'Overfit':<10} {'MIA Acc':<10} {'MIA Prec':<10} {'MIA Rec':<10}")
    print("-" * 120)
    
    for name, res in sorted(results.items()):
        model_type = 'MPC' if res['is_mpc'] else 'Plaintext'
        mia_acc = f"{res['mia_accuracy']:.2f}" if res['mia_accuracy'] is not None else 'N/A'
        mia_prec = f"{res['mia_precision']:.4f}" if res['mia_precision'] is not None else 'N/A'
        mia_rec = f"{res['mia_recall']:.4f}" if res['mia_recall'] is not None else 'N/A'
        train_acc = f"{res['train_accuracy']:.2f}" if res['train_accuracy'] is not None else 'N/A'
        overfit = f"{res['overfit_gap']:.2f}" if res['overfit_gap'] is not None else 'N/A'
        test_acc = f"{res['test_accuracy']:.2f}" if res['test_accuracy'] is not None else 'N/A'
        
        print(f"{name:<25} {model_type:<10} {test_acc:<10} {train_acc:<10} {overfit:<10} {mia_acc:<10} {mia_prec:<10} {mia_rec:<10}")
    
    print("=" * 120)


def save_results(results: dict, output_dir: str, filename: Optional[str] = None) -> str:
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary of results
        output_dir: Directory to save to
        filename: Optional filename (default: results_TIMESTAMP.json)
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Add metadata
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> dict:
    """Load results from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('results', data)
