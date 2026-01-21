import os
import time
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import crypten
import crypten.nn as cnn
from tqdm import tqdm


class EarlyStopping:    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
        return self.should_stop


def plaintext_train_model(model, train_loader, num_epochs=10, lr=0.001, device='cuda', 
                          save_dir=None, model_name='model', verbose=True, 
                          patience=10, min_delta=0.001):
    """
    Train a plaintext PyTorch model with early stopping.
    
    Saves intermediate checkpoints and a final checkpoint with _final.pt suffix.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        device: Training device
        save_dir: Directory to save checkpoints
        model_name: Name for checkpoint files
        verbose: Whether to show progress
        patience: Early stopping patience
        min_delta: Early stopping minimum delta
    
    Returns:
        model: Trained model
        history: Training history dict
        epochs_trained: Actual number of epochs trained (may be less due to early stopping)
    """
    model = model.to(device)
    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history = {
        'train_loss': [],
        'train_acc': []
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Calculate checkpoint epochs (every 1/5th of training)
    checkpoint_interval = max(1, num_epochs // 5)
    checkpoint_epochs = set(range(checkpoint_interval, num_epochs + 1, checkpoint_interval))
    checkpoint_epochs.add(num_epochs)

    epochs_trained = 0
    epoch_iter = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        epochs_trained = epoch + 1

        epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")
        
        # Save intermediate checkpoint
        current_epoch = epoch + 1
        if save_dir and current_epoch in checkpoint_epochs:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch{current_epoch}.pt")
            t.save(model.state_dict(), checkpoint_path)
            if verbose:
                tqdm.write(f"Checkpoint saved at epoch {current_epoch}")
        
        # Check early stopping
        if early_stopping(avg_loss):
            tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save final checkpoint with _final.pt suffix
    if save_dir:
        final_path = os.path.join(save_dir, f"{model_name}_epoch{epochs_trained}_final.pt")
        t.save(model.state_dict(), final_path)
        if verbose:
            tqdm.write(f"Final model saved to: {final_path}")

    return model, history, epochs_trained


def mpc_train_epoch(model, train_loader, optimizer, criterion, device, num_classes=10, verbose=True):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    batch_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc="Batch", leave=False, disable=not verbose)
    for batch_idx, (inputs, targets) in batch_iter:

        x_enc = crypten.cryptensor(inputs)
        y_one_hot = F.one_hot(targets, num_classes=num_classes).float() # Crypten requires one-hot labels
        y_enc = crypten.cryptensor(y_one_hot)
        
        optimizer.zero_grad()
        
        output_enc = model(x_enc)
        loss_enc = criterion(output_enc, y_enc)
        loss_enc.backward()
        optimizer.step()

        loss_val = loss_enc.get_plain_text().item()
        running_loss += loss_val
        
        # Decrypt predictions for accuracy
        output_plain = output_enc.get_plain_text()
        predictions = output_plain.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        batch_iter.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
    

def mpc_train_model(model, train_loader, num_epochs=10, lr=0.001, device='cpu', 
                    model_name='MpcModel', checkpoint_dir='./Checkpoints/MPC', verbose=True,
                    patience=10, min_delta=0.001):
    """
    Train an MPC model with early stopping.
    
    Returns:
        model: Trained model
        history: Training history dict
        epochs_trained: Actual number of epochs trained (may be less due to early stopping)
    """
    if not crypten.is_initialized():
        crypten.init()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.encrypt()
    model.train()

    optimizer = crypten.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = cnn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history = {
        'train_loss': [],
        'train_acc': []
    }

    # Calculate checkpoint epochs (every 1/5th of training)
    checkpoint_interval = max(1, num_epochs // 5)
    checkpoint_epochs = set(range(checkpoint_interval, num_epochs + 1, checkpoint_interval))
    # Always include final epoch
    checkpoint_epochs.add(num_epochs)

    if verbose:
        tqdm.write(f"Checkpoints stored at epochs: {sorted(checkpoint_epochs)}")
    start_time = time.time()

    epochs_trained = 0
    epoch_iter = tqdm(range(num_epochs), desc="MPC Training", disable=not verbose)
    
    for epoch in epoch_iter:
        train_loss, train_acc = mpc_train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device,
            verbose=verbose
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        epochs_trained = epoch + 1
        
        epoch_iter.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.2f}%")
        
        elapsed = time.time() - start_time
        current_epoch = epoch + 1

        if current_epoch in checkpoint_epochs:
            checkpoint_path = f'{checkpoint_dir}/{model_name}_epoch{current_epoch}.pt'
            crypten.save(model.state_dict(), checkpoint_path)
            if verbose:
                tqdm.write(f'Checkpoint saved at epoch {current_epoch}, at: {checkpoint_path}')
        
        # Check early stopping
        if early_stopping(train_loss):
            # Save final checkpoint on early stop
            final_path = f'{checkpoint_dir}/{model_name}_epoch{current_epoch}_final.pt'
            crypten.save(model.state_dict(), final_path)
            tqdm.write(f"Early stopping triggered at epoch {current_epoch}")
            break

    # Save final model if completed all epochs
    if epochs_trained == num_epochs:
        final_path = f'{checkpoint_dir}/{model_name}_epoch{num_epochs}_final.pt'
        crypten.save(model.state_dict(), final_path)
        if verbose:
            tqdm.write(f'Final model saved to: {final_path}')

    return model, history, epochs_trained
