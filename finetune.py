import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

from nnclr import NNCLR
from config import *

class FineTuningConfig:
    """Configuration class for fine-tuning parameters"""
    def __init__(
        self,
        freeze_encoder=False,
        encoder_lr=LEARNING_RATE,
        head_lr=LEARNING_RATE,
        num_epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        weight_decay=0.05
    ):
        self.freeze_encoder = freeze_encoder
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

def create_finetuning_dataloaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE):
    """
    Create dataloaders for finetuning phase.
    
    Args:
        X_train: Training data array
        y_train: Training labels array
        X_test: Test data array
        y_test: Test labels array
        batch_size: Batch size for dataloaders
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train for one epoch.
    
    Args:
        model: NNCLR model
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: torch device
        epoch: current epoch number
        num_epochs: total number of epochs
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        features = model.encoder(x)
        outputs = model.linear_probe(features)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss/(pbar.n+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Args:
        model: NNCLR model
        test_loader: Test data loader
        criterion: Loss function
        device: torch device
    
    Returns:
        tuple: (accuracy, average loss, confusion matrix)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            features = model.encoder(x)
            outputs = model.linear_probe(features)
            loss = criterion(outputs, y)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return accuracy, avg_loss, conf_matrix

def setup_model_for_finetuning(model, config: FineTuningConfig):
    """
    Setup model and optimizer for fine-tuning.
    
    Args:
        model: NNCLR model
        config: FineTuningConfig object
    
    Returns:
        optimizer: Configured optimizer
    """
    # Freeze or unfreeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = not config.freeze_encoder
    
    # Create parameter groups with different learning rates
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.linear_probe.parameters())
    
    param_groups = []
    
    # Add encoder params if not frozen
    if not config.freeze_encoder:
        param_groups.append({
            'params': encoder_params,
            'lr': config.encoder_lr
        })
    
    # Add head params
    param_groups.append({
        'params': head_params,
        'lr': config.head_lr
    })
    
    # Create optimizer
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    return optimizer

def finetune(
    pretrained_model_path,
    X_train,
    y_train,
    X_test,
    y_test,
    config: FineTuningConfig = None,
    device=None,
    save_path=None
):
    """
    Finetune pretrained NNCLR model.
    
    Args:
        pretrained_model_path: Path to pretrained model state dict
        X_train: Training data array
        y_train: Training labels array
        X_test: Test data array
        y_test: Test labels array
        config: FineTuningConfig object
        device: torch device (if None, will use CUDA if available)
        save_path: Path to save the finetuned model
        
    Returns:
        tuple: (finetuned model, training history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config is None:
        config = FineTuningConfig()
    
    print(f"Finetuning on device: {device}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Encoder frozen: {config.freeze_encoder}")
    print(f"Encoder LR: {config.encoder_lr}")
    print(f"Head LR: {config.head_lr}")
    
    # Create dataloaders
    train_loader, test_loader = create_finetuning_dataloaders(
        X_train, y_train, X_test, y_test, config.batch_size
    )
    
    # Load pretrained model
    model = NNCLR().to(device)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup model and optimizer for fine-tuning
    optimizer = setup_model_for_finetuning(model, config)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_acc': 0,
        'best_epoch': 0,
        'confusion_matrix': None
    }
    
    # Finetuning loop
    print("\nStarting finetuning...")
    for epoch in range(config.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config.num_epochs
        )
        
        # Evaluate
        test_acc, test_loss, conf_matrix = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > history['best_acc']:
            history['best_acc'] = test_acc
            history['best_epoch'] = epoch
            history['confusion_matrix'] = conf_matrix
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'config': config.__dict__
                }, save_path)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"Best Test Acc: {history['best_acc']:.2f}%")
    
    return model, history