import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *


def get_1d_sincos_pos_embed(embed_dim, positions):
    """Generate sinusoidal positional embeddings for 1D sequence positions.
    
    Args:
        embed_dim: Output dimension for each position
        positions: List of positions to encode
        
    Returns:
        Positional embeddings tensor of shape [len(positions), embed_dim]
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = omega / (embed_dim / 2.)
    omega = 1. / (10000**omega)
    
    pos = torch.as_tensor(positions, dtype=torch.float32)
    pos = pos.reshape(-1, 1)  # [len(positions), 1]
    omega = omega.reshape(1, -1)  # [1, embed_dim//2]
    
    out = torch.matmul(pos, omega)  # [len(positions), embed_dim//2]
    
    emb_sin = torch.sin(out)  # [len(positions), embed_dim//2]
    emb_cos = torch.cos(out)  # [len(positions), embed_dim//2]
    
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # [len(positions), embed_dim]
    return emb

class TimeSeriesEmbedding(nn.Module):
    """Converts time series segments into learned embeddings using convolutional layers."""
    
    def __init__(self, segment_size, n_channels, embed_dim, seq_length=96):
        """
        Args:
            segment_size: Number of timesteps per segment
            n_channels: Number of input channels
            embed_dim: Output embedding dimension
            seq_length: Length of input sequence (default=96 for HAR)
        """
        super().__init__()
        self.segment_size = segment_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # Calculate padding for each conv layer to maintain sequence length
        padding_1 = (11 - 1) // 2  # for kernel_size=11
        padding_2 = (7 - 1) // 2   # for kernel_size=7
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            # First conv layer: n_channels -> embed_dim//2
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=embed_dim//2,
                kernel_size=11,
                padding=padding_1
            ),
            nn.LayerNorm([embed_dim//2, seq_length]),
            nn.GELU(),
            
            # Second conv layer: embed_dim//2 -> embed_dim
            nn.Conv1d(
                in_channels=embed_dim//2,
                out_channels=embed_dim,
                kernel_size=7,
                padding=padding_2
            ),
            nn.LayerNorm([embed_dim, seq_length]),
            nn.GELU(),
        )
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # Max 1000 segments
        self._init_pos_embedding()
        
    def _init_pos_embedding(self):
        """Initialize positional embeddings with sinusoidal values."""
        positions = torch.arange(1000)
        pos_embed = get_1d_sincos_pos_embed(self.embed_dim, positions)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C] 
               where B=batch, T=timesteps, C=channels
        
        Returns:
            Embedded tensor of shape [B, N, D] 
            where N=num_segments, D=embed_dim
        """
        B, T, C = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"
        assert T == self.seq_length, f"Expected sequence length {self.seq_length}, got {T}"
        
        # Reshape for convolution [B, C, T]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = self.conv_layers(x)  # [B, embed_dim, T]
        
        # Reshape sequence into segments
        N = T // self.segment_size  # number of segments
        x = x.reshape(B, self.embed_dim, N, self.segment_size)
        
        # Pool within segments
        x = torch.mean(x, dim=3)  # [B, embed_dim, N]
        
        # Transpose to get [B, N, embed_dim]
        x = x.transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :N, :]
        
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        input_shape=INPUT_SHAPE,  # (seq_length, channels)
        segment_size=4,
        embed_dim=EMBEDDING_DIM,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Save config
        self.input_channels = input_shape[1]
        self.seq_length = input_shape[0]
        self.segment_size = segment_size
        self.num_segments = self.seq_length // segment_size
        self.embed_dim = embed_dim
        
        # Time series embedding
        self.embedding = TimeSeriesEmbedding(
            segment_size=segment_size,
            n_channels=self.input_channels,
            embed_dim=embed_dim
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights with small random values."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for NNCLR compatibility.
        
        Args:
            x: Input tensor of shape [B, T, C]
        
        Returns:
            Encoded representations of shape [B, embed_dim]
        """
        # Get embeddings [B, N, D]
        x = self.embedding(x)
        
        # Apply transformer [B, N, D]
        x = self.transformer(x)
        
        # Global average pooling for NNCLR compatibility [B, D]
        x = x.mean(dim=1)
        
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