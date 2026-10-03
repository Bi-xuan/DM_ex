"""Exercise 2/4: implement DDPM equations with mathematical indexing.

Arrays have length T+1; index 0 denotes clean data, and training uses 1..T.
Write the equations yourself, then compare with the references in README.md.
"""
import torch


def make_schedule(T: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Return beta, alpha, alpha_bar, and posterior_variance, each [T+1].

    beta[0] = 0; beta[1:] linearly spans 1e-4 to 0.02.
    alpha = 1-beta; alpha_bar is the cumulative product of alpha.
    posterior_variance[0] = 0, and for t >= 1:
        beta[t] * (1-alpha_bar[t-1]) / (1-alpha_bar[t]).
    In particular, alpha_bar[0]=1 and posterior_variance[1]=0.
    """
    raise NotImplementedError("Exercise 2: implement make_schedule")


def q_sample(x0, t, noise, schedule):
    """Directly corrupt clean points; return a tensor shaped like x0.

    x0 and noise: [B, 2]; t: [B] integer indices, possibly different per row.
    x_t = sqrt(alpha_bar[t])*x0 + sqrt(1-alpha_bar[t])*noise.
    Hint: gathered coefficients need shape [B, 1] for broadcasting.
    """
    raise NotImplementedError("Exercise 2: implement q_sample")


@torch.no_grad()
def p_sample(model, xt, t: int, schedule):
    """One reverse step: x_t -> x_(t-1), with the same shape and device.

    Make a [B] integer timestep tensor for the model.
    Compute the DDPM reverse mean from predicted noise (see README).
    Add Gaussian noise scaled by sqrt(posterior_variance[t]) only if t>1.
    """
    raise NotImplementedError("Exercise 4: implement p_sample")
