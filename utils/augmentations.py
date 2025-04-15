import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *

class Jittering(nn.Module):
    """
    Jittering the signal by adding random noise.
    Args:
        sigma: Standard deviation of the Gaussian distribution.
    """
    def __init__(self, sigma: float = JITTER_SIGMA):
        super().__init__()
        self.sigma = sigma

    def forward(self, signal):
        noise = torch.randn_like(signal) * self.sigma
        return signal + noise

class Scaling(nn.Module):
    """
    Scaling the signal by multiplying with random factors.
    Args:
        sigma: Standard deviation of the Gaussian distribution.
    """
    def __init__(self, sigma: float = SCALING_SIGMA):
        super().__init__()
        self.sigma = sigma

    def forward(self, signal):
        scale_factor = 1.0 + torch.randn_like(signal) * self.sigma
        return signal * scale_factor

class TimeWarping(nn.Module):
    """
    Time warping the signal by adding random displacement to random points.
    Args:
        sigma: Standard deviation of the Gaussian distribution.
        knot: Number of points to modify.
    """
    def __init__(self, sigma: float = TIMEWARPING_SIGMA, knot: int = TIMEWARPING_KNOT):
        super().__init__()
        self.sigma = sigma
        self.knot = knot

    def forward(self, signal):
        batch_size, seq_length, channels = signal.shape
        
        # Create base indices for all sequences in batch
        time_steps = torch.arange(seq_length, dtype=torch.float32, device=signal.device)
        time_steps = time_steps.expand(batch_size, -1)
        
        # Apply warping at random points
        for _ in range(self.knot):
            # Generate random points for each sequence in batch
            random_points = torch.randint(0, seq_length, (batch_size, 1), device=signal.device)
            
            # Generate random warping values
            random_warping = torch.randn(batch_size, 1, device=signal.device) * self.sigma
            
            # Create mask for the points after the random point
            mask = torch.arange(seq_length, device=signal.device).expand(batch_size, -1)
            mask = (mask >= random_points)
            
            # Apply warping
            time_steps = time_steps + mask.float() * random_warping
        
        # Clip values to valid range
        time_steps = torch.clamp(time_steps, 0, seq_length - 1)
        
        # Prepare grid for grid_sample
        # Scale time_steps to [-1, 1] range
        time_steps = 2 * time_steps / (seq_length - 1) - 1
        
        # Reshape time_steps for grid_sample [batch, height(1), width(seq_length), channels(1)]
        grid_x = time_steps.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_length, 1]
        grid_y = torch.zeros_like(grid_x)  # [batch, 1, seq_length, 1]
        grid = torch.cat([grid_x, grid_y], dim=-1)  # [batch, 1, seq_length, 2]
        
        # Reshape signal for grid_sample [batch, channels, height(1), width(seq_length)]
        signal_reshaped = signal.transpose(1, 2).unsqueeze(2)  # [batch, channels, 1, seq_length]
        
        # Apply interpolation
        warped_signal = F.grid_sample(
            signal_reshaped,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape back to original format [batch, seq_length, channels]
        warped_signal = warped_signal.squeeze(2).transpose(1, 2)
        
        return warped_signal

class TimeSeriesAugmenter(nn.Module):
    """
    Combines all augmentations in sequence.
    """
    def __init__(self, name="augmenter"):
        super().__init__()
        self.name = name
        self.jittering = Jittering()
        self.scaling = Scaling()
        self.timewarping = TimeWarping()
        
    def forward(self, x):
        x = self.jittering(x)
        x = self.scaling(x)
        x = self.timewarping(x)
        return x

def get_augmenter(name="augmenter"):
    """Factory function to create augmenter."""
    return TimeSeriesAugmenter(name=name)