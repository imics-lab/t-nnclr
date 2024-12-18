import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Encoder(nn.Module):
    """
    Encoder network for time series data.
    Architecture:
    - 2 Conv1D layers with ReLU activation
    - Dropout
    - MaxPool1D
    - Flatten
    - Dense layer to embedding dimension
    """
    def __init__(
        self,
        input_shape=INPUT_SHAPE,
        embedding_dim=EMBEDDING_DIM,
        conv_filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        pool_size=POOL_SIZE
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        
        # First Conv1D block
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[1],  # number of channels
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding='same'
        )
        
        # Second Conv1D block
        self.conv2 = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding='same'
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # MaxPool layer
        self.maxpool = nn.MaxPool1d(pool_size)
        
        # Calculate the size of flattened features
        self._build_shape_inference()
        
        # Dense layer for embedding
        self.fc = nn.Linear(self.flat_features, embedding_dim)
    
    def _build_shape_inference(self):
        """
        Calculate the size of flattened features after convolution and pooling.
        Used to determine the size of the fully connected layer.
        """
        # Create a dummy input to calculate shapes
        x = torch.randn(1, self.input_shape[1], self.input_shape[0])  # (batch, channels, length)
        
        # Pass through conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        # Calculate flattened size
        self.flat_features = x.numel() // x.size(0)
    
    def forward(self, x):
        """
        Forward pass of the encoder.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels)
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Transpose input to (batch_size, channels, sequence_length) for Conv1D
        x = x.transpose(1, 2)
        
        # First Conv block
        x = F.relu(self.conv1(x))
        
        # Second Conv block
        x = F.relu(self.conv2(x))
        
        # Dropout
        x = self.dropout(x)
        
        # MaxPool
        x = self.maxpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to embedding dimension
        x = F.relu(self.fc(x))
        
        return x


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Architecture:
    - 2-layer MLP with ReLU activation
    """
    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    """
    Prediction head for contrastive learning, used in asymmetric part of NNCLR.
    Architecture:
    - 2-layer MLP with ReLU activation in between
    """
    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM*2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def get_encoder():
    """Factory function to create encoder."""
    return Encoder()

def get_projection_head():
    """Factory function to create projection head."""
    return ProjectionHead()

def get_prediction_head():
    """Factory function to create prediction head."""
    return PredictionHead()