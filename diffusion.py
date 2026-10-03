"""Exercises 2 and 4: implement forward and reverse DDPM diffusion.

Replace the ``...`` expressions marked TODO. The guards stop at the first blank.
Use the schedule/forward exercises first; return to reverse diffusion after
implementing NoisePredictor.forward in model.py.

Indexing convention:
    x_0 is clean data; x_T is approximately standard Gaussian noise.
    Schedule tensors have length T+1, with index 0 reserved for clean data.
    Training samples integer timesteps 1..T, inclusive.

Useful Python/PyTorch syntax:
    schedule["alpha_bar"]     # Dictionary lookup by a string key.
    values[t]                # Scalar index OR a tensor of indices.
    values[1:]               # Slice from index 1 through the last element.
    values[:-1]              # Slice including all elements except the last.
    tensor.sqrt()            # Elementwise square root.
    tensor.unsqueeze(1)      # Insert a dimension at position 1.
    torch.cumprod(x, dim=0)   # Products along a chosen tensor dimension.

Do not confuse alpha_t (one step) with alpha_bar_t (the cumulative product).
Noise scale is a STANDARD DEVIATION: take the square root of its variance.
"""

import torch


def make_schedule(T: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Construct beta, alpha, alpha_bar and posterior_variance, each [T+1].

    All four tensors should be float32 and on device. This function has no
    learned parameters: the schedule is fixed before training.

    Equations:
        alpha_t = 1 - beta_t
        alpha_bar_t = product(alpha_s for s=1..t)
        posterior_variance_t = beta_t * (1-alpha_bar_(t-1)) / (1-alpha_bar_t)

    Boundary values:
        beta[0]=0; alpha[0]=1; alpha_bar[0]=1
        posterior_variance[0]=0; posterior_variance[1]=0
    """
    if T < 1:
        raise ValueError("T must be positive")

    # S1. Make T beta values increasing linearly from 1e-4 to 0.02.
    # Grammar: torch.linspace(start, end, steps, dtype=..., device=...)
    # `steps` is the number of values, not the distance between them.
    # Include both endpoints; request float32 on the supplied device.
    beta_values = torch.linspace(1e-4, 0.02, T, dtype=torch.float32, device=device)  # TODO S1: a [T] tensor of beta values.
    if beta_values is Ellipsis:
        raise NotImplementedError("TODO S1: create the linear beta values")

    # S2. Prepend one zero so mathematical timestep t matches array index t.
    # Grammar: torch.zeros(size, dtype=..., device=...)
    # Grammar: torch.cat(sequence_of_tensors, dim=0)
    # The sequence can be a list or tuple. Concatenate a [1] zero tensor with
    # beta_values [T]. cat joins existing dimensions; stack adds a new dimension.
    beta = torch.cat([torch.zeros(1, dtype=torch.float32, device=device), beta_values])  # TODO S2: construct beta with shape [T+1] and beta[0]=0.
    if beta is Ellipsis:
        raise NotImplementedError("TODO S2: prepend beta's timestep-zero entry")

    # S3. Compute one-step signal-retention coefficients.
    # Grammar: scalar - tensor; arithmetic applies to every element.
    alpha = 1 - beta  # TODO S3: translate alpha_t = 1-beta_t.
    if alpha is Ellipsis:
        raise NotImplementedError("TODO S3: compute alpha")

    # S4. Compute cumulative products, not a cumulative sum.
    # Grammar: torch.cumprod(input_tensor, dim=0)
    # Example: cumulative products of [1, 0.8, 0.5] are [1, 0.8, 0.4].
    alpha_bar = torch.cumprod(alpha, dim=0)  # TODO S4: obtain all cumulative alpha products.
    if alpha_bar is Ellipsis:
        raise NotImplementedError("TODO S4: compute alpha_bar")

    # S5. Compute the posterior variance only for timesteps 1..T.
    # Grammar: tensor_a * tensor_b / tensor_c; use parentheses for (1 - ...).
    # beta[1:] and alpha_bar[1:] refer to the current timesteps.
    # alpha_bar[:-1] refers to their corresponding PREVIOUS timesteps.
    # These slices all have length T. Apply the equation in the docstring.
    # Do not evaluate it at index 0: that would involve division by zero.
    variance_values = beta[1:] * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])  # TODO S5: compute posterior variances with shape [T].
    if variance_values is Ellipsis:
        raise NotImplementedError("TODO S5: compute the posterior variance values")

    posterior_variance = torch.zeros_like(beta)
    posterior_variance[1:] = variance_values

    # A dictionary groups named tensors. Callers retrieve them with schedule[key].
    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "posterior_variance": posterior_variance,
    }


def q_sample(x0, t, noise, schedule):
    """Corrupt clean points directly at arbitrary noise levels.

    Inputs:
        x0, noise: [B, 2], floating-point tensors on the same device.
        t: [B], integer indices; different rows can have different timesteps.
        schedule: dictionary returned by make_schedule.
    Output:
        xt: [B, 2], on the same device as x0.

    Equation:
        x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise

    The caller supplies noise so its exact value can also be the training target.
    Do not draw another random noise tensor inside this function.
    """
    alpha_bar = schedule["alpha_bar"]

    # Q1. Look up the cumulative coefficient separately for each batch row.
    # Grammar: lookup_table[index_tensor]
    # A [B] integer tensor indexing a [T+1] table produces a [B] result.
    # Use alpha_bar, not the one-step alpha values.
    selected_alpha_bar = alpha_bar[t]  # TODO Q1: gather the coefficients at t.
    if selected_alpha_bar is Ellipsis:
        raise NotImplementedError("TODO Q1: look up alpha_bar at the batch timesteps")

    # Q2. Reshape the gathered coefficients for broadcasting over coordinates.
    # Grammar: tensor.unsqueeze(dim)
    # Insert dimension 1 to turn [B] into [B, 1]. This broadcasts over [B, 2].
    # Keeping shape [B] can either fail or silently give wrong results when B=2.
    coefficients = selected_alpha_bar.unsqueeze(1)  # TODO Q2: convert selected_alpha_bar to shape [B, 1].
    if coefficients is Ellipsis:
        raise NotImplementedError("TODO Q2: add the coordinate-broadcast dimension")

    # Q3. Translate the equation above using coefficients, x0 and noise.
    # Grammar: tensor.sqrt(), multiplication (*), and addition (+).
    # Both the signal coefficient and noise variance require square roots.
    # The original data is scaled as well as the noise; this is not x0 + noise.
    xt = coefficients.sqrt() * x0 + (1 - coefficients).sqrt() * noise  # TODO Q3: combine the scaled signal and supplied noise.
    if xt is Ellipsis:
        raise NotImplementedError("TODO Q3: apply the forward corruption equation")

    # Self-check: t=0 returns x0, since alpha_bar[0]=1.
    return xt


@torch.no_grad()
def p_sample(model, xt, t: int, schedule):
    """Generate x_(t-1) from x_t using one reverse DDPM step.

    Here t is ONE Python integer shared by the batch (unlike q_sample's [B] t).
    xt and the returned tensor have shape [B, 2]. The model expects a [B]
    integer timestep tensor, so construct one before calling it.

    Equations:
        mu = (x_t - beta_t/sqrt(1-alpha_bar_t)*predicted_noise)/sqrt(alpha_t)
        x_previous = mu + sqrt(posterior_variance_t)*z, for t>1
        x_previous = mu, for t=1
    z is freshly sampled standard Gaussian noise.

    @torch.no_grad() is a decorator: it disables gradient tracking inside this
    function. Sampling needs no backward pass. The caller also sets model.eval().
    """
    if not 1 <= t < len(schedule["beta"]):
        raise ValueError("The reverse timestep must be between 1 and T")

    # P1. Repeat the integer timestep for every point in the batch.
    # Grammar: torch.full(size, fill_value, dtype=..., device=...)
    # `size` is a tuple; `(xt.shape[0],)` describes a one-dimensional batch.
    # Use dtype=torch.long (integer indices) and device=xt.device.
    batch_t = torch.full((xt.shape[0],), t, dtype=torch.long, device=xt.device)  # TODO P1: construct a [B] tensor filled with t.
    if batch_t is Ellipsis:
        raise NotImplementedError("TODO P1: build the model's batch timestep tensor")

    # P2. Ask the trained model to predict the noise.
    # Grammar: model(input_tensor, timestep_tensor)
    # Calling an nn.Module invokes its forward method; use the module call.
    # Pass xt and batch_t. The result must have the same shape as xt.
    predicted_noise = model(xt, batch_t)  # TODO P2: obtain the model's prediction.
    if predicted_noise is Ellipsis:
        raise NotImplementedError("TODO P2: predict the noise")

    # Scalar tensor lookups broadcast automatically over all points/coordinates.
    beta_t = schedule["beta"][t]
    alpha_t = schedule["alpha"][t]
    alpha_bar_t = schedule["alpha_bar"][t]

    # P3. Translate the reverse-mean equation in the docstring.
    # Grammar: parentheses group arithmetic; tensor.sqrt() takes square roots.
    # Divide the WHOLE corrected xt expression by sqrt(alpha_t).
    # The multiplier of predicted noise depends on beta_t and alpha_bar_t.
    mean = (xt - beta_t / (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_t.sqrt()  # TODO P3: calculate the DDPM reverse mean.
    if mean is Ellipsis:
        raise NotImplementedError("TODO P3: calculate the reverse mean")

    if t > 1:
        # P4. Sample fresh Gaussian randomness for this reverse step.
        # Grammar: torch.randn_like(reference_tensor)
        # Keep its shape, dtype and device consistent with xt.
        z = torch.randn_like(xt)  # TODO P4: sample standard normal noise shaped like xt.
        if z is Ellipsis:
            raise NotImplementedError("TODO P4: sample reverse-step Gaussian noise")

        # P5. Convert posterior variance into a noise standard deviation.
        # Grammar: schedule[key][index] retrieves a scalar tensor; .sqrt()
        # operates on it. Variance and standard deviation are not interchangeable.
        noise_scale = schedule["posterior_variance"][t].sqrt()  # TODO P5: retrieve and square-root the variance at t.
        if noise_scale is Ellipsis:
            raise NotImplementedError("TODO P5: calculate the reverse noise scale")

        # P6. Add the scaled randomness to the reverse mean.
        # Grammar: tensor_a + scalar_tensor * tensor_b
        x_previous = mean + noise_scale * z  # TODO P6: combine mean, noise_scale and z.
        if x_previous is Ellipsis:
            raise NotImplementedError("TODO P6: combine the reverse mean and noise")
        return x_previous

    # The final step adds no new randomness. No extra TODO is needed here.
    return mean
