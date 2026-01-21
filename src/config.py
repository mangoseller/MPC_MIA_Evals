import os
from dataclasses import dataclass, field, fields

def setup_dirs(base_dir=None):
    """
    Setup and create experiment directories.
    Checkpoints are stored one level up from the code base directory.
    """
    if base_dir is None:
        # Get the directory containing this file, then go one level up
        code_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(code_dir)
    
    checkpoint_dir = os.path.join(base_dir, 'Checkpoints')
    
    DIRS = {
        'plaintext': os.path.join(checkpoint_dir, 'plaintext'),
        'MPC':       os.path.join(checkpoint_dir, 'MPC'),
        'shadow_models':  os.path.join(checkpoint_dir, 'shadow_models'),
        'attack_models':  os.path.join(checkpoint_dir, 'attack_models'),
    }
    # Create all directories
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    return DIRS


@dataclass
class ExperimentConfig:
    cnn_epochs: int = 80
    mlp_epochs: int = 120
    lenet_epochs: int = 40
    mpc_cnn_epochs: int = 80
    mpc_mlp_epochs: int = 120
    mpc_lenet_epochs: int = 40
    attack_epochs: int = 30
    num_shadow_models: int = 9
    target_train_size: int = 10000
    batch_size: int = 128
    mpc_batch_size: int = 32
    learning_rate: float = 1e-2
    shadow_pool_ratio: float = 0.5
    seed: int = 42
    num_workers: int = 2
    # Early stopping parameters (only for target models)
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    def get_plaintext_epochs(self, model_name: str) -> int:
        if 'CNN' in model_name:
            return self.cnn_epochs
        elif 'MLP' in model_name:
            return self.mlp_epochs
        elif 'LeNet' in model_name:
            return self.lenet_epochs
        raise ValueError("Model type not found!")
    
    def get_mpc_epochs(self, model_name: str) -> int:
        if 'CNN' in model_name:
            return self.mpc_cnn_epochs
        elif 'MLP' in model_name:
            return self.mpc_mlp_epochs
        elif 'LeNet' in model_name:
            return self.mpc_lenet_epochs
        raise ValueError("Model type not found!")
 
    def get_field_names(self):
        return [f.name for f in fields(self)]
    
    def get_field_type(self, field_name: str):
        for f in fields(self):
            if f.name == field_name:
                return f.type
        return None
    
    def set_field(self, field_name: str, value_str: str) -> tuple[bool, str]:
        # Set a field value from a string.
    
        if field_name not in self.get_field_names():
            return False, f"Setting '{field_name}' not found"
        
        field_type = self.get_field_type(field_name)
        
        try:
            if field_type == int:
                value = int(value_str)
            elif field_type == float:
                value = float(value_str)
            elif field_type == bool:
                value = value_str.lower() in ('true', '1', 'yes')
            elif field_type == str:
                value = value_str
            else:
                value = value_str
            
            setattr(self, field_name, value)
            return True, f"Set {field_name} = {value}"
        except ValueError as e:
            return False, f"Invalid value for {field_name}: {value_str} (expected {field_type.__name__})"
    
    def display_table(self):
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"{'Setting':<30} {'Value':<25}")
        print("-" * 60)
        
        # Group settings logically
        groups = {
            'Plaintext Training': ['cnn_epochs', 'mlp_epochs', 'lenet_epochs'],
            'MPC Training': ['mpc_cnn_epochs', 'mpc_mlp_epochs', 'mpc_lenet_epochs'],
            'Shadow & Attack': ['attack_epochs', 'num_shadow_models'],
            'Data': ['target_train_size', 'shadow_pool_ratio', 'seed'],
            'Training': ['batch_size', 'mpc_batch_size', 'learning_rate', 'num_workers'],
            'Early Stopping (Target Only)': ['early_stopping_patience', 'early_stopping_min_delta'],
        }
        
        for group_name, setting_names in groups.items():
            print(f"\n  {group_name}:")
            for name in setting_names:
                value = getattr(self, name)
                print(f"    {name:<28} {value}")
        
        print("\n  Note: Shadow models train for the same epochs as their target.")
        print("=" * 60)
