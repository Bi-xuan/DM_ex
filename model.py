"""Exercise 3: guide a small MLP to predict noise in 2D points.

The network architecture is provided. Complete the five TODO expressions in
forward(); replace each ``...`` with your code. The guards identify unfinished
steps, just as in data.py and diffusion.py.

Python class grammar:
    class NoisePredictor(nn.Module):  # Inherit PyTorch's model behavior.
    def __init__(self, ...):          # Initialize a new model instance.
    self.attribute = value           # Store something on that instance.
    def forward(self, xt, t):         # Define what the model does to inputs.

`self` refers to the current instance and is supplied automatically:
    model = NoisePredictor()  # Calls __init__ with default settings.
    prediction = model(xt, t) # nn.Module routes this call to forward.

Prefer model(xt, t) to calling model.forward(xt, t) directly; the module call
also handles PyTorch hooks. Parameters belong to the network; the timestep
normalization below has no trainable parameters.
"""

import torch
from torch import nn


class NoisePredictor(nn.Module):
    def __init__(self, T=1000, hidden=128):
        """Create a model with two hidden layers and a two-coordinate output.

        T is the same total diffusion-step count used in make_schedule.
        hidden controls the width of the intermediate feature vectors.
        """
        # Initialize nn.Module so layers/parameters are registered correctly.
        # Grammar: super() accesses behavior inherited from the parent class.
        super().__init__()
        if T < 1 or hidden < 1:
            raise ValueError("T and hidden must be positive")

        # Keep T on the instance so forward() can use it on every call.
        self.T = T

        # nn.Sequential applies its modules in the listed order.
        # nn.Linear(in_features, out_features) learns a weight matrix and bias.
        # For a batch: [B, in_features] -> [B, out_features].
        # SiLU is an elementwise nonlinear activation; it preserves the shape.
        # Shape path: [B, 3] -> [B, hidden] -> [B, hidden] -> [B, 2].
        # Three input features = noisy x coordinate, noisy y coordinate, time.
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        # No final activation: noise predictions can be positive or negative
        # and should not be restricted to a bounded interval.

    def forward(self, xt, t):
        """Return predicted noise [B, 2] from noisy points and timesteps.

        Inputs (already on the same device as the model):
            xt: [B, 2], floating-point noisy coordinates.
            t:  [B], integer timestep for each point; normally between 1 and T.

        Intermediate shape/dtype guide:
            t_float       [B]    floating, matching xt.dtype
            t_normalized  [B]    floating, divided by T
            t_column      [B, 1] floating
            model_input   [B, 3] floating: [noisy_x, noisy_y, normalized_t]
            predicted_noise [B, 2] floating

        This model estimates the unscaled Gaussian noise used to construct xt.
        It gets neither the clean x0 nor the true noise as an input.
        Diffusion time here describes noise level, not developmental age.
        """

        # M1. Convert integer timesteps to xt's floating-point dtype.
        # Grammar: tensor.to(dtype=desired_dtype)
        # Obtain the desired dtype with xt.dtype; .to(...) returns a tensor.
        # Keep the original t unchanged: it may still be needed for indexing.
        # Conversion does not add/remove dimensions, so the shape stays [B].
        t_float = t.to(dtype=xt.dtype)  # TODO M1: convert t to the same dtype as xt.
        if t_float is Ellipsis:
            raise NotImplementedError("TODO M1: convert timesteps to floating point")

        # M2. Scale time to a convenient numeric range.
        # Grammar: tensor / scalar; access the saved step count with self.T.
        # E.g. timestep 500 with T=1000 should become 0.5.
        # Use regular division (/), not floor division (//).
        t_normalized = t_float / self.T  # TODO M2: divide the floating timesteps by T.
        if t_normalized is Ellipsis:
            raise NotImplementedError("TODO M2: normalize timesteps")

        # M3. Turn the timestep vector into a feature column.
        # Grammar: tensor.unsqueeze(dim)
        # Insert dimension 1 to transform [B] into [B, 1].
        # This differs from unsqueeze(0), which would give [1, B].
        t_column = t_normalized.unsqueeze(1)  # TODO M3: reshape normalized time into [B, 1].
        if t_column is Ellipsis:
            raise NotImplementedError("TODO M3: create a timestep feature column")

        # M4. Join each point with its own time feature.
        # Grammar: torch.cat(sequence_of_tensors, dim=chosen_dimension)
        # The sequence can be a tuple or list. Join xt and t_column along
        # the FEATURE dimension (1), producing [B, 3]. Keep the batch size B.
        # dim=0 joins batch rows; stack inserts a new dimension instead.
        model_input = torch.cat([xt, t_column], dim=1)  # TODO M4: concatenate coordinates and time.
        if model_input is Ellipsis:
            raise NotImplementedError("TODO M4: combine noisy coordinates and time")

        # M5. Apply the provided network to the prepared input.
        # Grammar: module(input_tensor); use the stored module self.net.
        # The final Linear layer maps the hidden features to two predictions.
        # Do not detach the result or disable gradients here: train.py needs
        # loss.backward() to reach the model's weights through this computation.
        predicted_noise = self.net(model_input)  # TODO M5: pass model_input through the network.
        if predicted_noise is Ellipsis:
            raise NotImplementedError("TODO M5: evaluate the noise-prediction network")

        return predicted_noise
