import os
from dataclasses import dataclass, fields


def setup_dirs(base_dir=None, dataset_name="cifar10", seed=42):
    """
    Setup experiment directories scoped by dataset and seed.

    Layout:
        Checkpoints/{dataset}/seed_{seed}/plaintext/
        Checkpoints/{dataset}/seed_{seed}/shadow_models/
        Checkpoints/{dataset}/seed_{seed}/attack_models/
        results/{dataset}/
    """
    if base_dir is None:
        code_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(code_dir)

    run_dir = os.path.join(base_dir, "Checkpoints", dataset_name, f"seed_{seed}")

    DIRS = {
        "base": base_dir,
        "plaintext": os.path.join(run_dir, "plaintext"),
        "shadow_models": os.path.join(run_dir, "shadow_models"),
        "attack_models": os.path.join(run_dir, "attack_models"),
        "results": os.path.join(base_dir, "results", dataset_name),
    }
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    print(f"Checkpoints: {run_dir}")
    return DIRS


@dataclass
class ExperimentConfig:
    cnn_epochs: int = 100
    mlp_epochs: int = 150
    lenet_epochs: int = 60
    mpc_cnn_epochs: int = 100
    mpc_mlp_epochs: int = 150
    mpc_lenet_epochs: int = 60
    attack_epochs: int = 30
    num_shadow_models: int = 22
    target_train_size: int = 20000
    batch_size: int = 128
    mpc_batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    shadow_pool_ratio: float = 0.5
    seed: int = 42
    num_workers: int = 2
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0005

    def get_plaintext_epochs(self, model_name: str) -> int:
        if "CNN" in model_name:
            return self.cnn_epochs
        elif "MLP" in model_name:
            return self.mlp_epochs
        elif "LeNet" in model_name:
            return self.lenet_epochs
        raise ValueError(f"Unknown model type in: {model_name}")

    def get_mpc_epochs(self, model_name: str) -> int:
        if "CNN" in model_name:
            return self.mpc_cnn_epochs
        elif "MLP" in model_name:
            return self.mpc_mlp_epochs
        elif "LeNet" in model_name:
            return self.mpc_lenet_epochs
        raise ValueError(f"Unknown model type in: {model_name}")

    # ── field introspection for interactive config ──

    def get_field_names(self):
        return [f.name for f in fields(self)]

    def get_field_type(self, field_name: str):
        for f in fields(self):
            if f.name == field_name:
                return f.type
        return None

    def set_field(self, field_name: str, value_str: str) -> tuple[bool, str]:
        if field_name not in self.get_field_names():
            return False, f"Setting '{field_name}' not found"

        field_type = self.get_field_type(field_name)
        try:
            if field_type == int:
                value = int(value_str)
            elif field_type == float:
                value = float(value_str)
            elif field_type == bool:
                value = value_str.lower() in ("true", "1", "yes")
            else:
                value = value_str
            setattr(self, field_name, value)
            return True, f"Set {field_name} = {value}"
        except ValueError:
            return False, f"Invalid value for {field_name}: {value_str}"

    def display_table(self):
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)

        groups = {
            "Plaintext Training": ["cnn_epochs", "mlp_epochs", "lenet_epochs"],
            "MPC Epoch Budget": ["mpc_cnn_epochs", "mpc_mlp_epochs", "mpc_lenet_epochs"],
            "Shadow & Attack": ["attack_epochs", "num_shadow_models"],
            "Data": ["target_train_size", "shadow_pool_ratio", "seed"],
            "Training": ["batch_size", "mpc_batch_size", "learning_rate", "weight_decay", "num_workers"],
            "Early Stopping (Target)": ["early_stopping_patience", "early_stopping_min_delta"],
        }

        for group_name, setting_names in groups.items():
            print(f"\n  {group_name}:")
            for name in setting_names:
                value = getattr(self, name)
                print(f"    {name:<28} {value}")

        print("\n  Note: Shadow models train for the same epochs as their target.")
        print("  Note: Shadow models are shared by both basic MIA and LiRA.")
        print("  Note: MPC models are obtained by weight transfer, not training.")
        print("=" * 60)
