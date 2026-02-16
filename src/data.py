import os
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def partition_dataset_for_mia(full_dataset, target_train_size, shadow_pool_ratio=0.5, seed=42):
    """Partition dataset into disjoint pools for target model and shadow models."""
    np.random.seed(seed)
    dataset_size = len(full_dataset)
    all_indices = np.random.permutation(dataset_size)

    if target_train_size >= dataset_size:
        raise ValueError(
            f"target_train_size ({target_train_size}) >= dataset size ({dataset_size})")

    target_train_indices = all_indices[:target_train_size]
    remaining_indices = all_indices[target_train_size:]

    shadow_pool_size = int(len(remaining_indices) * shadow_pool_ratio)
    shadow_pool_indices = remaining_indices[:shadow_pool_size]
    target_test_indices = remaining_indices[shadow_pool_size:]

    if len(target_test_indices) < 1000:
        raise ValueError(
            f"Only {len(target_test_indices)} non-member test samples remaining. "
            f"Reduce target_train_size or shadow_pool_ratio.")

    print(f"Target train: {len(target_train_indices)}  |  "
          f"Target test (non-members): {len(target_test_indices)}  |  "
          f"Shadow pool: {len(shadow_pool_indices)}")

    return target_train_indices, target_test_indices, shadow_pool_indices


def train_shadow_models(
    num_shadows, model_class, full_dataset, shadow_pool_indices,
    model_name, base_dir, num_epochs, num_classes=10,
    device="cuda", verbose=True, lr=1e-2,
):
    """
    Train shadow models on the shadow pool.

    Shadow models train for exactly *num_epochs* (matching the target model).
    No early stopping is applied.
    """
    shadow_models = []
    shadow_data_indices = []
    shadow_pool_size = len(shadow_pool_indices)
    split_size = shadow_pool_size // 2

    shadow_save_dir = os.path.join(base_dir, model_name)
    os.makedirs(shadow_save_dir, exist_ok=True)

    print(f"Training {num_shadows} shadow models ({shadow_pool_size} pool, "
          f"{num_epochs} epochs, {num_classes} classes)...")

    shadow_iter = tqdm(range(num_shadows), desc="Shadow Models", disable=not verbose)
    for i in shadow_iter:
        perm = np.random.permutation(shadow_pool_size)
        train_local = perm[:split_size]
        test_local = perm[split_size:]

        train_indices = shadow_pool_indices[train_local]
        test_indices = shadow_pool_indices[test_local]

        save_path = os.path.join(shadow_save_dir, f"shadow_{i}.pt")

        if os.path.exists(save_path):
            # Resume: load already-trained shadow model
            shadow_model = model_class(num_classes=num_classes)
            shadow_model.load_state_dict(t.load(save_path, map_location="cpu"))
            shadow_iter.set_postfix(model=f"{i+1}/{num_shadows}", status="loaded")
        else:
            train_subset = Subset(full_dataset, train_indices)
            train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)

            shadow_model = model_class(num_classes=num_classes).to(device)
            shadow_iter.set_postfix(model=f"{i+1}/{num_shadows}", status="training")

            shadow_model = _train_shadow_model(
                shadow_model, train_loader, num_epochs, lr=lr, device=device, verbose=False,
            )

            t.save(shadow_model.state_dict(), save_path)
            shadow_model.cpu()
            t.cuda.empty_cache()

        shadow_models.append(shadow_model)
        shadow_data_indices.append((train_indices, test_indices))

    return shadow_models, shadow_data_indices


def _train_shadow_model(model, train_loader, num_epochs, lr, device, verbose=False):
    """Train a shadow model for exactly num_epochs (no early stopping)."""
    model = model.to(device)
    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch_iter = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for _ in epoch_iter:
        model.train()
        running_loss = 0.0
        correct = total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        epoch_iter.set_postfix(
            loss=f"{running_loss / len(train_loader):.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )
    return model


def regenerate_shadow_indices(num_shadows, shadow_pool_indices, seed=42):
    """Reproduce shadow train/test splits deterministically."""
    np.random.seed(seed)
    pool_size = len(shadow_pool_indices)
    split = pool_size // 2

    indices = []
    for _ in range(num_shadows):
        perm = np.random.permutation(pool_size)
        train_idx = shadow_pool_indices[perm[:split]]
        test_idx = shadow_pool_indices[perm[split:]]
        indices.append((train_idx, test_idx))
    return indices
