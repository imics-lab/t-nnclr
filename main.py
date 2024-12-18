import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
import logging
from datetime import datetime

from pretrain import pretrain
from finetune import finetune, FineTuningConfig
from config import *
from data_loader import load_and_preprocess_har_data

def setup_logging(experiment_dir):
    """Setup logging configuration."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = experiment_dir / 'experiment.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_cross_validation_experiment(
    data_dir='./data',
    experiment_dir=None,
    pretrain_epochs=PRETRAIN_EPOCHS,
    finetune_epochs=FINETUNE_EPOCHS,
    freeze_encoder=False,
    encoder_lr=LEARNING_RATE,
    head_lr=LEARNING_RATE
):
    """
    Run NNCLR experiment with Leave-One-Subject-Out cross validation.
    
    Args:
        data_dir: Directory containing the data files
        experiment_dir: Directory to save experiment results
        pretrain_epochs: Number of pretraining epochs
        finetune_epochs: Number of finetuning epochs
        freeze_encoder: Whether to freeze encoder during fine-tuning
        encoder_lr: Learning rate for encoder during fine-tuning
        head_lr: Learning rate for classification head during fine-tuning
    
    Returns:
        dict: Results dictionary containing accuracies and confusion matrices
    """
    # Setup experiment directory and logging
    if experiment_dir is None:
        experiment_dir = Path.cwd() / 'experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        experiment_dir = Path(experiment_dir)
    
    logger = setup_logging(experiment_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    X_unlabeled, X_labeled, y, subjects = load_and_preprocess_har_data(data_dir, logger)
    
    # Create fine-tuning configuration
    finetune_config = FineTuningConfig(
        freeze_encoder=freeze_encoder,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        num_epochs=finetune_epochs
    )
    
    # Perform pretraining once using only unlabeled data
    logger.info("Starting pretraining phase...")
    pretrain_save_path = experiment_dir / 'pretrained_model.pt'
    
    pretrained_model, pretrain_history = pretrain(
        X_unlabeled=X_unlabeled,
        num_epochs=pretrain_epochs,
        device=device,
        save_path=pretrain_save_path
    )
    
    # Save pretraining history
    np.save(experiment_dir / 'pretrain_history.npy', pretrain_history)
    
    # Setup cross-validation
    cv = LeaveOneGroupOut()
    splits = list(cv.split(X_labeled, y, subjects))
    logger.info(f"Running Leave-One-Subject-Out CV with {len(splits)} folds")
    
    # Rest of the code remains unchanged
    results = {
        'fold_accuracies': [],
        'confusion_matrices': [],
        'finetune_histories': []
    }
    
    # Run cross-validation
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold_dir = experiment_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        
        logger.info(f"\nStarting Fold {fold + 1}/{len(splits)}")
        
        # Split data for this fold
        X_train = X_labeled[train_idx]
        y_train = y[train_idx]
        X_test = X_labeled[test_idx]
        y_test = y[test_idx]
        
        # Finetuning phase
        logger.info("Starting finetuning phase...")
        finetune_save_path = fold_dir / 'finetuned_model.pt'
        
        # Fine-tune using pretrained weights
        model, finetune_history = finetune(
            pretrained_model_path=pretrain_save_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=finetune_config,
            device=device,
            save_path=finetune_save_path
        )
        
        # Store results for this fold
        results['fold_accuracies'].append(finetune_history['best_acc'])
        results['confusion_matrices'].append(finetune_history['confusion_matrix'])
        results['finetune_histories'].append(finetune_history)
        
        logger.info(f"Fold {fold + 1} - Best Test Accuracy: {finetune_history['best_acc']:.2f}%")
    
    # Calculate and log final results
    accuracies = np.array(results['fold_accuracies'])
    cumulative_confusion = sum(results['confusion_matrices'])
    
    logger.info("\nFinal Results:")
    logger.info(f"Mean Accuracy: {accuracies.mean():.2f}% +/- {accuracies.std():.2f}%")
    logger.info("\nCumulative Confusion Matrix:")
    logger.info("\n" + str(cumulative_confusion))
    
    # Save final results
    np.save(experiment_dir / 'accuracies.npy', accuracies)
    np.save(experiment_dir / 'confusion_matrices.npy', results['confusion_matrices'])
    np.save(experiment_dir / 'cumulative_confusion.npy', cumulative_confusion)
    
    # Save summary statistics
    with open(experiment_dir / 'summary.txt', 'w') as f:
        f.write("Experiment Configuration:\n")
        f.write(f"Freeze encoder: {freeze_encoder}\n")
        f.write(f"Encoder learning rate: {encoder_lr}\n")
        f.write(f"Head learning rate: {head_lr}\n")
        f.write(f"Pretrain epochs: {pretrain_epochs}\n")
        f.write(f"Finetune epochs: {finetune_epochs}\n\n")
        f.write(f"Mean Accuracy: {accuracies.mean():.2f}% +/- {accuracies.std():.2f}%\n")
        f.write(f"Individual Fold Accuracies: {accuracies.tolist()}\n")
        f.write(f"\nCumulative Confusion Matrix:\n{cumulative_confusion}")
    
    return results

if __name__ == "__main__":
    # Set paths
    data_dir = Path.cwd() / 'data'
    experiment_dir = Path.cwd() / 'experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run experiment with specific configuration
    results = run_cross_validation_experiment(
        data_dir=data_dir,
        experiment_dir=experiment_dir,
        pretrain_epochs=PRETRAIN_EPOCHS,
        finetune_epochs=FINETUNE_EPOCHS,
        freeze_encoder=False,
        encoder_lr=LEARNING_RATE,
        head_lr=LEARNING_RATE
    )
    
    print("\nExperiment completed!")
    print(f"Results saved to {experiment_dir}")