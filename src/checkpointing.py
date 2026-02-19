import os
import glob
import re
import torch as t
import crypten
from attack import AttackNet


def check_plaintext_target_exists(name, dirs):
    pattern = os.path.join(dirs["plaintext"], f"{name}_epoch*_final.pt")
    finals = glob.glob(pattern)
    if finals:
        return True, sorted(finals)[-1]
    return False, None


def check_mpc_target_exists(name, dirs):
    pattern = os.path.join(dirs.get("MPC", ""), f"{name}_epoch*_final.pt")
    finals = glob.glob(pattern)
    if finals:
        return True, sorted(finals)[-1]
    return False, None


def find_latest_intermediate_checkpoint(name, checkpoint_dir):
    """Find latest non-final checkpoint. Returns (found, path, epoch)."""
    pattern = os.path.join(checkpoint_dir, f"{name}_epoch*.pt")
    all_ckpts = glob.glob(pattern)
    intermediate = [p for p in all_ckpts if not p.endswith("_final.pt")]

    best_epoch, best_path = -1, None
    for path in intermediate:
        m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(path))
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch, best_path = epoch, path

    if best_path is not None:
        return True, best_path, best_epoch
    return False, None, None


def check_shadow_models_exist(name, dirs, num_shadows):
    shadow_dir = os.path.join(dirs["shadow_models"], name)
    if not os.path.exists(shadow_dir):
        return False, []
    paths = [os.path.join(shadow_dir, f"shadow_{i}.pt") for i in range(num_shadows)]
    return all(os.path.exists(p) for p in paths), paths


def check_attack_model_exists(arch_key, dirs):
    path = os.path.join(dirs["attack_models"], f"attack_{arch_key}.pt")
    return os.path.exists(path), path


def extract_epochs_from_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        return None
    m = re.search(r"_epoch(\d+)_final\.pt$", os.path.basename(checkpoint_path))
    return int(m.group(1)) if m else None


# ── loaders ───────────────────────────────────────────────────────────────

def load_plaintext_model(model_class, path, device, num_classes=10):
    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(t.load(path, map_location=device))
    print(f"  Loaded plaintext: {path}")
    return model


def load_mpc_model(model_class, path, num_classes=10):
    if not crypten.is_initialized():
        crypten.init()
    model = model_class(num_classes=num_classes)
    model.encrypt()
    model.load_state_dict(crypten.load(path))
    print(f"  Loaded MPC: {path}")
    return model


def load_shadow_models(model_class, paths, device, num_classes=10):
    models = []
    for path in paths:
        m = model_class(num_classes=num_classes)
        m.load_state_dict(t.load(path, map_location="cpu"))
        models.append(m)
    print(f"  Loaded {len(models)} shadow models (on CPU)")
    return models


def load_attack_model(path, device, num_classes=10):
    """Load attack model. Handles both old (raw state_dict) and new (dict with input_dim) formats."""
    checkpoint = t.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim", num_classes)
        model = AttackNet(input_dim=input_dim).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Legacy format: raw state_dict
        model = AttackNet(input_dim=num_classes).to(device)
        model.load_state_dict(checkpoint)

    print(f"  Loaded attack model: {path}")
    return model


def check_all_training_complete(model_names, dirs, num_shadows):
    """
    Check if all training artifacts (plaintext targets, shadow models,
    attack models) exist for every architecture — i.e. phases 1–3 are done.

    Returns True only if every model has a final checkpoint, all shadows,
    and an attack model on disk.
    """
    for name in model_names:
        # Plaintext target
        exists, _ = check_plaintext_target_exists(name, dirs)
        if not exists:
            return False
        # Shadow models
        exists, _ = check_shadow_models_exist(name, dirs, num_shadows)
        if not exists:
            return False
        # Attack model
        arch = name.replace("PlainText", "")
        exists, _ = check_attack_model_exists(arch, dirs)
        if not exists:
            return False
    return True


def get_attack_model_for_architecture(arch_key, attack_models, dirs, device, num_classes=10):
    if arch_key in attack_models:
        return attack_models[arch_key]
    exists, path = check_attack_model_exists(arch_key, dirs)
    if exists:
        return load_attack_model(path, device, num_classes)
    return None
