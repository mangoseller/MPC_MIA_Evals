import os
import glob
import re
import torch as t
import crypten
from attack import AttackNet

def check_plaintext_target_exists(name, dirs):
    """
    Check if a COMPLETE plaintext target model checkpoint exists.
    Complete checkpoints are marked with _final.pt suffix.
    
    Returns:
        (exists: bool, path: str or None)
    """
    # Look for final checkpoint (could be at any epoch due to early stopping)
    pattern = os.path.join(dirs['plaintext'], f"{name}_epoch*_final.pt")
    final_checkpoints = glob.glob(pattern)
    
    if final_checkpoints:
        # Return the most recent final checkpoint
        final_path = sorted(final_checkpoints)[-1]
        return True, final_path
    
    return False, None

def check_mpc_target_exists(name, dirs):
    """
    Check if a COMPLETE MPC checkpoint exists for a model.
    Complete checkpoints are marked with _final.pt suffix.
    
    Returns:
        (exists: bool, path: str or None)
    """
    pattern = os.path.join(dirs['MPC'], f"{name}_epoch*_final.pt")
    final_checkpoints = glob.glob(pattern)
    
    if final_checkpoints:
        final_path = sorted(final_checkpoints)[-1]
        return True, final_path
    
    return False, None


def find_latest_intermediate_checkpoint(name, checkpoint_dir):
    """
    Find the latest intermediate (non-final) checkpoint for a model.

    Searches for checkpoint files matching {name}_epoch{N}.pt (excluding _final.pt)
    and returns the one with the highest epoch number.

    Args:
        name: Model name (e.g., 'PlainTextCNN_Sigmoid')
        checkpoint_dir: Directory containing checkpoints

    Returns:
        (found: bool, path: str or None, epoch: int or None)
    """
    pattern = os.path.join(checkpoint_dir, f"{name}_epoch*.pt")
    all_checkpoints = glob.glob(pattern)

    # Filter out final checkpoints
    intermediate = [p for p in all_checkpoints if not p.endswith('_final.pt')]

    if not intermediate:
        return False, None, None

    # Extract epochs and find the latest
    best_epoch = -1
    best_path = None
    for path in intermediate:
        filename = os.path.basename(path)
        match = re.search(r'_epoch(\d+)\.pt$', filename)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = path

    if best_path is not None:
        return True, best_path, best_epoch

    return False, None, None


def check_shadow_models_exist(name, dirs, num_shadows):
    shadow_dir = os.path.join(dirs['shadow_models'], name)
    if not os.path.exists(shadow_dir):
        return False, [] 
    paths = [os.path.join(shadow_dir, f"shadow_{i}.pt") for i in range(num_shadows)]
    all_exist = all(os.path.exists(p) for p in paths)
    return all_exist, paths


def check_attack_model_exists(arch_key, dirs):
    path = os.path.join(dirs['attack_models'], f"attack_{arch_key}.pt")
    return os.path.exists(path), path


def extract_epochs_from_checkpoint(checkpoint_path):
    """
    Extract the number of epochs from a checkpoint filename.
    
    Examples:
        "PlainTextCNN_Sigmoid_epoch20_final.pt" -> 20
        "MpcMLP_Tanh_epoch80_final.pt" -> 80
    
    Returns:
        int: Number of epochs, or None if not found
    """
    if checkpoint_path is None:
        return None
    
    filename = os.path.basename(checkpoint_path)
    match = re.search(r'_epoch(\d+)_final\.pt$', filename)
    if match:
        return int(match.group(1))
    return None

def load_plaintext_model(model_class, path, device):
    model = model_class(num_classes=10).to(device)
    model.load_state_dict(t.load(path, map_location=device))
    print(f"Loaded plaintext model from: {path}")
    return model

def load_mpc_model(model_class, path):
    if not crypten.is_initialized():
        crypten.init()
    
    model = model_class(num_classes=10)
    model.encrypt()
    state_dict = crypten.load(path)
    model.load_state_dict(state_dict)
    print(f"Loaded MPC model from: {path}")
    return model


def load_shadow_models(model_class, paths, device):
    shadow_models = []
    for path in paths:
        model = model_class(num_classes=10).to(device)
        model.load_state_dict(t.load(path, map_location=device))
        shadow_models.append(model)
    print(f"Loaded {len(shadow_models)} shadow models")
    return shadow_models


def load_attack_model(path, device):
    model = AttackNet().to(device)
    model.load_state_dict(t.load(path, map_location=device))
    print(f"Loaded attack model from: {path}")
    return model

def get_attack_model_for_architecture(arch_key, attack_models, dirs, device):
    """
    Get the attack model for a given architecture.
    For MPC models, this reuses the plaintext attack model.
    
    Args:
        arch_key: Architecture key (e.g., 'CNN_Sigmoid', 'MLP_Tanh')
        attack_models: Dict of already-loaded attack models
        dirs: Directory configuration
        device: Device to load model to
    
    Returns:
        attack_model: The attack model for this architecture, or None if not found
    """
    # Check if already loaded in memory
    if arch_key in attack_models:
        return attack_models[arch_key]
    
    # Try to load from disk
    attack_exists, attack_path = check_attack_model_exists(arch_key, dirs)
    if attack_exists:
        return load_attack_model(attack_path, device)
    
    return None
