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
        # Input is the target model's logit vector (size 10 for CIFAR-10) 
        self.fc1 = nn.Linear(input_dim, 64)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1) # 0 Non-Member, 1 Member
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def prepare_attack_dataset(shadow_models, shadow_indices, full_dataset, device='cuda', verbose=True):
    """Generate (logit vector, membership label) pairs from shadow models."""

    X_attack = []
    y_attack = []
    
    with t.no_grad():
        model_iter = tqdm(enumerate(shadow_models), total=len(shadow_models), desc="Preparing Attack Dataset", disable=not verbose)
        for i, model in model_iter:
            model.eval()
            train_idx, test_idx = shadow_indices[i]
            train_set = set(train_idx.tolist()) if hasattr(train_idx, 'tolist') else set(train_idx)
            
            # Combine train and test indices for this shadow model
            all_shadow_idx = np.concatenate([train_idx, test_idx])
            shadow_subset = Subset(full_dataset, all_shadow_idx)
            shadow_loader = DataLoader(shadow_subset, batch_size=128, shuffle=False, num_workers=0)
            
            # Get predictions for shadow model's data
            all_preds = []
            for inputs, _ in shadow_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = F.softmax(outputs, dim=1)
                all_preds.append(preds.cpu())

            all_preds = t.cat(all_preds)
            
            # Label based on membership
            for j, idx in enumerate(all_shadow_idx):
                pred_vector = all_preds[j]
                label = 1.0 if idx in train_set else 0.0
                
                X_attack.append(pred_vector)
                y_attack.append(label)
                
    X_attack = t.stack(X_attack)
    y_attack = t.tensor(y_attack).unsqueeze(1)
    
    return X_attack, y_attack


def train_attack_model(X_attack, y_attack, save_path=None, epochs=20, device='cuda', verbose=True):
    """
    Train the attack model for exactly the specified number of epochs.
    No early stopping is applied to attack models.
    
    Args:
        X_attack: Attack features (softmax outputs from shadow models)
        y_attack: Attack labels (membership labels)
        save_path: Path to save the trained model
        epochs: Number of epochs to train
        device: Training device
        verbose: Whether to show progress
    
    Returns:
        attack_model: Trained attack model
    """
    attack_model = AttackNet().to(device)
    optimizer = t.optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    
    dataset = t.utils.data.TensorDataset(X_attack, y_attack)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    epoch_iter = tqdm(range(epochs), desc="Attack Model Training", disable=not verbose)
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = attack_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t.save(attack_model.state_dict(), save_path)
        if verbose:
            tqdm.write(f"Saved attack model to: {save_path}") 
             
    return attack_model

def evaluate_mia_attack(target_model, attack_model, train_loader, test_loader, device, is_mpc=False, verbose=True):
    """
    Evaluates MIA on either Plaintext or MPC models.
    For MPC models (even decrypted), inputs MUST be CrypTensors.
    """
    # Only call eval() on standard PyTorch models
    if not is_mpc:
        target_model.eval()
    attack_model.eval()
    
    def get_target_preds(loader, is_member, desc=""):
        preds = []
        labels = []

        was_decrypted = False
        if is_mpc and hasattr(target_model, 'encrypted') and not target_model.encrypted:
            target_model.encrypt() # Toggle flag so it accepts CrypTensors
            was_decrypted = True
        
        loader_iter = tqdm(loader, desc=desc, leave=False, disable=not verbose)
        for inputs, _ in loader_iter:
            if is_mpc:
                with t.no_grad():
                    x_input = crypten.cryptensor(inputs)
                    output_enc = target_model(x_input)
                    # Use get_plain_text() to get logits for the attack model
                    batch_preds = F.softmax(output_enc.get_plain_text(), dim=1)
                    del x_input, output_enc # Free cryptensors
                gc.collect()
            else:
                # Standard PyTorch Path
                inputs = inputs.to(device)
                with t.no_grad():
                    outputs = target_model(inputs)
                    batch_preds = F.softmax(outputs, dim=1)
            
            preds.append(batch_preds.cpu())
            labels.extend([1.0 if is_member else 0.0] * inputs.size(0))
            
        return t.cat(preds), t.tensor(labels).unsqueeze(1)
    
    member_preds, member_labels = get_target_preds(train_loader, is_member=True, desc="Members")
    non_member_preds, non_member_labels = get_target_preds(test_loader, is_member=False, desc="Non-members")
    
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
    
def evaluate_accuracy_loss(model, test_loader, criterion, device, verbose=True):
    """Evaluate a plaintext PyTorch model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loader_iter = tqdm(test_loader, desc="Evaluating", leave=False, disable=not verbose)
    with t.no_grad():
        for inputs, targets in loader_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def evaluate_accuracy_loss_mpc(model, test_loader, criterion, num_classes=10, verbose=True):
    running_loss = 0.0
    correct = 0
    total = 0

    # Ensure model is encrypted; CrypTen layers (cnn.*) are designed 
    # to process CrypTensors, and they check the 'encrypted' flag.

    was_encrypted = getattr(model, 'encrypted', False)
    if not was_encrypted:
        model.encrypt()

    loader_iter = tqdm(test_loader, desc="Evaluating MPC", leave=False, disable=not verbose)
    for inputs, targets in loader_iter:
        # Wrap inputs
        x_enc = crypten.cryptensor(inputs)
        
        with t.no_grad():
            output_enc = model(x_enc)
            # Decrypt outputs to calculate standard loss/acc
            outputs = output_enc.get_plain_text()
        
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    # Restore original state
    if not was_encrypted:
        model.decrypt()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
