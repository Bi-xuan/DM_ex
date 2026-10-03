"""Exercise 1: draw equally weighted points from four Gaussian clusters."""
import torch


def sample_data(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return float32 points of shape [batch_size, 2] on device.

    Centers: (-1, -1), (-1, 1), (1, -1), (1, 1).
    Choose centers uniformly and add independent Gaussian noise with std 0.1.
    Hint: use torch.randint for center indices and torch.randn for noise.
    """
    raise NotImplementedError("Exercise 1: implement sample_data in data.py")
