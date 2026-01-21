import os
import numpy as np
import torch as t
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from training import plaintext_train_model


def partition_dataset_for_mia(full_dataset, target_train_size, shadow_pool_ratio=0.5, seed=42):
    
# Partition dataset into disjoint pools for target model and shadow models

    np.random.seed(seed)
    dataset_size = len(full_dataset)
    all_indices = np.random.permutation(dataset_size)
    
    # Allocate target training set
    target_train_indices = all_indices[:target_train_size]
    remaining_indices = all_indices[target_train_size:]
    
    # Split remaining data between target test set and shadow pool
    shadow_pool_size = int(len(remaining_indices) * shadow_pool_ratio)
    shadow_pool_indices = remaining_indices[:shadow_pool_size]

   # Slice the test indices to match target_train_size
    target_test_indices = remaining_indices[shadow_pool_size : shadow_pool_size + target_train_size]
    
    print(f"Target model training set size: {len(target_train_indices)}")
    print(f"Target model testing set size:  {len(target_test_indices)}")
    print(f"Shadow model training pool size:  {len(shadow_pool_indices)}")
    
    return target_train_indices, target_test_indices, shadow_pool_indices

def train_shadow_models(num_shadows, model_class, full_dataset, shadow_pool_indices, 
                        model_name, base_dir, num_epochs=10, device='cuda', verbose=True,
                        patience=10, min_delta=0.001):
    """
    Trains shadow models on the shadow pool, which is disjoint from the target model's training data.
    
    Args:
        num_shadows: Number of shadow models to train
        model_class: Model class constructor
        full_dataset: Full training dataset
        shadow_pool_indices: Indices for shadow model training pool
        model_name: Name for saving checkpoints
        base_dir: Base directory for saving shadow models
        num_epochs: Number of epochs to train (should match target model's max epochs)
        device: Training device
        verbose: Whether to show progress
        patience: Early stopping patience
        min_delta: Early stopping minimum delta
    
    Returns:
        shadow_models: List of trained shadow models
        shadow_data_indices: List of (train_indices, test_indices) tuples
    """

    shadow_models = []
    shadow_data_indices = []
    shadow_pool_size = len(shadow_pool_indices)
    split_size = shadow_pool_size // 2  # 50% in, 50% out 
    
    shadow_save_dir = os.path.join(base_dir, model_name)
    os.makedirs(shadow_save_dir, exist_ok=True)

    print(f"Training {num_shadows} shadow models on pool of {shadow_pool_size} samples...")
    print(f"Shadow models will train for {num_epochs} epochs (matching target model config)")

    shadow_iter = tqdm(range(num_shadows), desc="Shadow Models", disable=not verbose)
    for i in shadow_iter:
        # Random permutation within the shadow pool (shadow datasets may overlap)
        perm = np.random.permutation(shadow_pool_size)
        train_local_indices = perm[:split_size]
        test_local_indices = perm[split_size:]
        
        # Map local indices back to full dataset indices
        train_indices = shadow_pool_indices[train_local_indices]
        test_indices = shadow_pool_indices[test_local_indices]
        
        train_subset = Subset(full_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
        
        shadow_model = model_class(num_classes=10).to(device)
        
        shadow_iter.set_postfix(model=f"{i+1}/{num_shadows}")
        
        # Shadow models save directly with simple naming (no _final suffix needed)
        shadow_model, _, _ = plaintext_train_model(
            shadow_model, 
            train_loader, 
            num_epochs=num_epochs, 
            lr=1e-3, 
            device=device,
            save_dir=None,  # Don't use the new save format for shadows
            model_name=f"shadow_{i}",
            verbose=False,  # Suppress inner progress to avoid clutter
            patience=patience,
            min_delta=min_delta
        )
    
        save_path = os.path.join(shadow_save_dir, f"shadow_{i}.pt")
        t.save(shadow_model.state_dict(), save_path)

        shadow_models.append(shadow_model)
        shadow_data_indices.append((train_indices, test_indices))
        
    return shadow_models, shadow_data_indices


def regenerate_shadow_indices(num_shadows, shadow_pool_indices, seed=42):
    """
    Regenerate the shadow model train/test indices using the same random seed.
    This is necessary when loading shadow models to reconstruct which samples
    were used for training vs testing each shadow model.
    """
    np.random.seed(seed)
    shadow_pool_size = len(shadow_pool_indices)
    split_size = shadow_pool_size // 2
    
    shadow_data_indices = []
    for i in range(num_shadows):
        perm = np.random.permutation(shadow_pool_size)
        train_local_indices = perm[:split_size]
        test_local_indices = perm[split_size:]
        
        train_indices = shadow_pool_indices[train_local_indices]
        test_indices = shadow_pool_indices[test_local_indices]
        
        shadow_data_indices.append((train_indices, test_indices))
    
    return shadow_data_indices
