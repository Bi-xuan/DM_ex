"""Exercise 3: a small noise predictor; no U-Net is needed for 2D data."""
import torch
from torch import nn


class NoisePredictor(nn.Module):
    def __init__(self, T=1000, hidden=128):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, xt, t):
        """Return predicted noise [B, 2] from xt [B, 2] and t [B].

        Convert t to xt's floating dtype, divide by T, reshape to [B, 1],
        concatenate it with xt, and feed the resulting [B, 3] into self.net.
        The model never receives clean points or the true sampled noise.
        """
        raise NotImplementedError("Exercise 3: implement NoisePredictor.forward")
