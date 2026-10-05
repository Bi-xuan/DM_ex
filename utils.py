"""Small device, plotting, and reproducibility helpers."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Allocate a GPU and activate its PyTorch environment.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use an MPS-enabled PyTorch environment on a supported Mac, or --device cpu.")
    if name == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    if name == "mps":
        return torch.device("mps", 0)
    return torch.device(name)


def scatter_panels(panels, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, (title, points) in zip(axes[0], panels):
        points = points.detach().cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.35)
        ax.set(title=title, xlim=(-4, 4), ylim=(-4, 4), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
