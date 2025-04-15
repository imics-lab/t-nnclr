import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from nnclr import NNCLR
from utils.config import *

def create_pretraining_dataloader(X_unlabeled, batch_size=BATCH_SIZE):
    """
    Create dataloader for pretraining phase.
    
    Args:
        X_unlabeled: Unlabeled data array
        batch_size: Batch size for dataloader
    
    Returns:
        DataLoader: Dataloader for unlabeled data
    """
    # Convert to torch tensor
    X_unlabeled = torch.FloatTensor(X_unlabeled)
    
    # Create dataset
    unlabeled_dataset = TensorDataset(X_unlabeled)
    
    # Create dataloader
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return unlabeled_loader

def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs):
    """
    Train for one epoch.
    
    Args:
        model: NNCLR model
        train_loader: DataLoader for unlabeled data
        optimizer: PyTorch optimizer
        device: torch device
        epoch: current epoch number
        num_epochs: total number of epochs
    
    Returns:
        float: average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch in pbar:
        # Get batch data
        x = batch[0].to(device)
        
        # Forward pass
        loss = model(x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss/(pbar.n+1):.4f}"
        })
    
    return total_loss / len(train_loader)

def pretrain(
    X_unlabeled,
    num_epochs=PRETRAIN_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device=None,
    save_path=None
):
    """
    Pretrain NNCLR model using only unlabeled data.
    
    Args:
        X_unlabeled: Unlabeled data array
        num_epochs: Number of pretraining epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: torch device (if None, will use CUDA if available)
        save_path: Path to save the pretrained model
        
    Returns:
        tuple: (trained model, training history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Pretraining on device: {device}")
    print(f"Unlabeled data shape: {X_unlabeled.shape}")
    
    # Create dataloader
    train_loader = create_pretraining_dataloader(X_unlabeled, batch_size)
    
    # Create model and optimizer
    model = NNCLR().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Training history
    history = {
        'loss': []
    }
    
    # Training loop
    print("\nStarting pretraining...")
    for epoch in range(num_epochs):
        loss = train_epoch(
            model, train_loader, optimizer, device, epoch, num_epochs
        )
        
        history['loss'].append(loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Loss: {loss:.4f}")
    
    # Save model if path provided
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, save_path)
        print(f"\nModel saved to {save_path}")
    
    return model, history

if __name__ == "__main__":
    # This section is for testing the pretraining script independently
    from pathlib import Path
    from data_loader import load_and_preprocess_har_data
    
    data_dir = Path.cwd() / 'data'
    X_unlabeled, _, _, _ = load_and_preprocess_har_data(data_dir)
    
    save_path = Path.cwd() / 'models' / 'pretrained_nnclr.pt'
    save_path.parent.mkdir(exist_ok=True)
    
    model, history = pretrain(
        X_unlabeled=X_unlabeled,
        save_path=save_path
    )
    print("Pretraining completed!")