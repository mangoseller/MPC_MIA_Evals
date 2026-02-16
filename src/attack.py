import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import crypten
from tqdm import tqdm
import gc


class AttackNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        return self.sigmoid(self.fc3(x))


def prepare_attack_dataset(shadow_models, shadow_indices, full_dataset,
                           device="cuda", verbose=True):
    """Generate (softmax vector, membership label) pairs from shadow models."""
    X_attack, y_attack = [], []

    with t.no_grad():
        for i, model in tqdm(enumerate(shadow_models), total=len(shadow_models),
                             desc="Preparing Attack Dataset", disable=not verbose):
            model.to(device).eval()
            train_idx, test_idx = shadow_indices[i]
            train_set = set(train_idx.tolist() if hasattr(train_idx, "tolist") else train_idx)

            all_idx = np.concatenate([train_idx, test_idx])
            loader = DataLoader(Subset(full_dataset, all_idx), batch_size=128,
                                shuffle=False, num_workers=0)

            all_preds = []
            for inputs, _ in loader:
                inputs = inputs.to(device)
                preds = F.softmax(model(inputs), dim=1)
                all_preds.append(preds.cpu())

            all_preds = t.cat(all_preds)
            for j, idx in enumerate(all_idx):
                X_attack.append(all_preds[j])
                y_attack.append(1.0 if idx in train_set else 0.0)

            model.cpu()
            t.cuda.empty_cache()

    return t.stack(X_attack), t.tensor(y_attack).unsqueeze(1)


def train_attack_model(X_attack, y_attack, num_classes=10, save_path=None,
                       epochs=20, device="cuda", verbose=True):
    """Train the binary attack classifier."""
    attack_model = AttackNet(input_dim=num_classes).to(device)
    optimizer = t.optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    loader = DataLoader(t.utils.data.TensorDataset(X_attack, y_attack),
                        batch_size=64, shuffle=True)

    epoch_iter = tqdm(range(epochs), desc="Attack Model Training", disable=not verbose)
    for _ in epoch_iter:
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(attack_model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_iter.set_postfix(loss=f"{epoch_loss / len(loader):.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t.save({"state_dict": attack_model.state_dict(), "input_dim": num_classes},
               save_path)
        if verbose:
            tqdm.write(f"Saved attack model to: {save_path}")

    return attack_model
