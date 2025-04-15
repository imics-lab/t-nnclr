import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.augmentations import get_augmenter
from models.encoder import get_encoder, get_projection_head, get_prediction_head
from utils.config import *

class NNCLR(nn.Module):
    """
    Nearest-Neighbor Contrastive Learning of Representations (NNCLR).
    Adapted for time series data, using only unlabeled data for pretraining.
    """
    def __init__(
        self,
        temperature=TEMPERATURE,
        queue_size=QUEUE_SIZE,
    ):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # Create components
        self.contrastive_augmenter = get_augmenter("contrastive_augmenter")
        self.encoder = get_encoder()
        self.projection_head = get_projection_head()
        self.prediction_head = get_prediction_head()
        
        # Initialize feature queue
        self.register_buffer("feature_queue", 
                           torch.randn(queue_size, EMBEDDING_DIM))
        self.feature_queue = F.normalize(self.feature_queue, dim=1)
        
        # Queue counter
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Create linear probe for supervised fine-tuning
        self.linear_probe = nn.Linear(EMBEDDING_DIM, N_CLASSES)

    @torch.no_grad()
    def _update_queue(self, features):
        """Update feature queue with current batch."""
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace features
        if ptr + batch_size > self.queue_size:
            # Handle queue overflow
            remaining = self.queue_size - ptr
            self.feature_queue[ptr:] = features[:remaining]
            self.feature_queue[:batch_size-remaining] = features[remaining:]
            ptr = batch_size-remaining
        else:
            self.feature_queue[ptr:ptr + batch_size] = features
            ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr

    def _find_nn(self, projections):
        """Find nearest neighbor for each projection in queue."""
        # Normalize projections and queue
        projections = F.normalize(projections, dim=1)
        queue = F.normalize(self.feature_queue, dim=1)
        
        # Compute similarities
        similarities = torch.mm(projections, queue.T)
        
        # Find nearest neighbors
        _, nn_idx = similarities.max(dim=1)
        
        # Gather nearest neighbors
        nn_projections = self.feature_queue[nn_idx]
        
        return nn_projections

    def _compute_contrastive_loss(self, nn_features, predictions):
        """Compute contrastive loss between nearest neighbors and predictions."""
        # Normalize features
        nn_features = F.normalize(nn_features, dim=1)
        predictions = F.normalize(predictions, dim=1)
        
        # Compute similarities
        similarities = torch.mm(predictions, nn_features.T) / self.temperature
        
        # Create labels (diagonal is positive pairs)
        labels = torch.arange(similarities.shape[0], device=similarities.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(similarities, labels)
        
        return loss

    def forward(self, x):
        """
        Forward pass for training.
        Args:
            x: Input data tensor
        """
        # Apply augmentations
        aug1 = self.contrastive_augmenter(x)
        aug2 = self.contrastive_augmenter(x)
        
        # Get features and projections
        features1 = self.encoder(aug1)
        features2 = self.encoder(aug2)
        
        projections1 = self.projection_head(features1)
        projections2 = self.projection_head(features2)
        
        # Get predictions
        predictions1 = self.prediction_head(projections1)
        predictions2 = self.prediction_head(projections2)
        
        # Find nearest neighbors
        nn1 = self._find_nn(projections1)
        nn2 = self._find_nn(projections2)
        
        # Update queue with current batch projections
        self._update_queue(projections1.detach())
        
        # Compute contrastive loss
        loss1 = self._compute_contrastive_loss(nn1, predictions2)
        loss2 = self._compute_contrastive_loss(nn2, predictions1)
        contrastive_loss = (loss1 + loss2) / 2
        
        return contrastive_loss

    def encode(self, x):
        """Encode data using trained encoder."""
        return self.encoder(x)

    def predict(self, x):
        """Get predictions using linear probe."""
        features = self.encoder(x)
        return self.linear_probe(features)