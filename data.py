"""Exercise 1: sample points from four equally weighted Gaussian clusters.

Only edit the five lines marked TODO. Replace each ``...`` with your code.
Python calls ``...`` the Ellipsis object; here it marks an unfinished exercise.
The guards below stop at the first blank and tell you which step to complete.

Useful Python syntax:
    variable = expression             # Assign a value to a name.
    function(argument, keyword=value) # Pass positional and keyword arguments.
    [item_1, item_2]                   # A list; lists can contain other lists.
    (length,)                         # A one-element tuple: the comma matters.
    table[row_indices]                # Select rows using an index tensor.

The type hints in the function signature document inputs and outputs:
``batch_size: int`` means an integer is expected; ``-> torch.Tensor`` describes
the returned value. Python does not enforce these annotations automatically.
"""

import torch


def sample_data(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return float32 points of shape [batch_size, 2] on the requested device.

    Each row is one [x, y] point. Choose among these centers uniformly:
        (-1, -1), (-1, 1), (1, -1), (1, 1).
    Add independent Gaussian noise with standard deviation 0.1 to each
    coordinate. Sample fresh center indices and noise on every function call.

    Shape guide (B means batch_size):
        centers          [4, 2]  float32
        center_indices   [B]     integer indices
        selected_centers [B, 2]  float32
        noise            [B, 2]  float32, standard normal before scaling
        points           [B, 2]  float32
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    num_clusters = 4
    noise_std = 0.1

    # Step 1: create the table of four cluster centers.
    # Grammar: torch.tensor(data, dtype=..., device=...)
    # `data` should be a nested list: [[x1, y1], [x2, y2], ...].
    # Use dtype=torch.float32 and the function's `device` argument.
    # The dtype and device are keyword arguments, written name=value.
    centers = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32, device=device)  # TODO 1: create a [4, 2] tensor containing the centers.
    if centers is Ellipsis:
        raise NotImplementedError("TODO 1: create the centers tensor")

    # Step 2: choose one cluster index independently for each point.
    # Grammar: torch.randint(low, high, size, device=...)
    # `low` is included; `high` is excluded. Choose indices from 0 to 3.
    # `size` is a tuple of dimensions. For B indices, use a one-element tuple.
    # `(batch_size,)` is a tuple; `(batch_size)` is just an integer.
    # torch.randint returns integer tensors (torch.int64) by default.
    center_indices = torch.randint(0, num_clusters, (batch_size,), device=device)  # TODO 2: draw a [B] tensor of uniform cluster indices.
    if center_indices is Ellipsis:
        raise NotImplementedError("TODO 2: sample cluster indices")

    # Step 3: select the chosen center for every point.
    # Grammar: table[index_tensor]
    # A [B] integer index tensor selects B rows from a [4, 2] table.
    # Repeated indices repeat rows; you do not need a Python for-loop.
    selected_centers = centers[center_indices]  # TODO 3: gather the selected centers into [B, 2].
    if selected_centers is Ellipsis:
        raise NotImplementedError("TODO 3: select the center rows")

    # Step 4: sample independent standard Gaussian noise.
    # Grammar option A: torch.randn(*size, dtype=..., device=...)
    # `*size` means dimensions may be passed separately, e.g. rows, columns.
    # Grammar option B: torch.randn_like(reference_tensor)
    # randn_like inherits the reference tensor's shape, dtype and device.
    # Use standard normal noise here: mean 0, standard deviation 1.
    # randn samples Gaussian values; rand samples uniform values instead.
    noise = torch.randn(batch_size,2, dtype=torch.float32, device=device)  # TODO 4: draw standard normal noise with shape [B, 2].
    if noise is Ellipsis:
        raise NotImplementedError("TODO 4: sample Gaussian noise")

    # Step 5: scale the noise and move it to the selected centers.
    # Grammar: tensor_a + scalar * tensor_b
    # Multiplication happens before addition. Multiplying a tensor by a scalar
    # scales every element; adding equally shaped tensors works elementwise.
    # Use selected_centers, noise_std and noise. Do not modify centers in place.
    points = selected_centers + noise_std*noise  # TODO 5: combine centers and noise to obtain the final points.
    if points is Ellipsis:
        raise NotImplementedError("TODO 5: combine centers and scaled noise")

    return points
