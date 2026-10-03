# DM_ex

A Day-2 diffusion exercise for learning DDPM on four Gaussian clusters, using
VS Code and optionally an A100 cluster. The mathematical functions are exercises;
the training, sampling, plotting and command-line harnesses are provided.
`NotImplementedError` is expected until you complete the relevant exercises.

## Setup

Use Python 3.9 or newer. Prefer your cluster's documented PyTorch environment or
container. For your own environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA, choose a PyTorch installation compatible with your cluster's NVIDIA
driver using the [official installer](https://pytorch.org/get-started/locally/).
The CPU path works for learning and checking the equations.

Open this folder in VS Code. Install Python and Python Debugger, then select your
interpreter with **Python: Select Interpreter**. `.vscode/launch.json` includes
CPU forward-check and five-update training debug configurations.

With [Remote SSH](https://code.visualstudio.com/docs/remote/ssh), code and Python
run on the connected host. A login-node connection does not allocate a GPU.
Run GPU commands inside an allocation; the VS Code debugger stays on its connected
host unless you use your cluster's supported compute-node development setup.

## Exercises, in order

### 1. Generate data: `data.py`

- [ ] Implement `sample_data`: choose uniformly among (-1,-1), (-1,1), (1,-1),
      (1,1), and add Gaussian noise with standard deviation 0.1.
- [ ] Return float32 tensors of shape `[B, 2]` on the requested device.
- [ ] Run `python check_forward.py --part data` and inspect `outputs/checks/data.png`.

### 2. Forward diffusion: `diffusion.py`

- [ ] Implement `make_schedule` and `q_sample`.
- [ ] Use mathematical indexing: arrays of length `T+1`, clean data at index 0,
      and training indices `1..T`.
- [ ] Implement the following equations in PyTorch:

    alpha_t = 1 - beta_t
    alpha_bar_t = product(alpha_s, s=1..t)
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon

- [ ] Support different timesteps for each row of a batch. Gathered coefficients
      should broadcast from `[B, 1]` to `[B, 2]`.
- [ ] Run `python check_forward.py --part forward`.
- [ ] Inspect `outputs/checks/forward.png`. Explain why terminal points are nearly Gaussian.
- [ ] Derive the direct corruption formula by expanding two consecutive steps.

The checker compares direct corruption with independently simulated one-step
corruption using empirical means and variances. Matching individual samples is
not required. Statistical checks use a fixed seed and finite-sample tolerances.

### 3. Noise predictor: `model.py`

- [ ] Implement `NoisePredictor.forward`: concatenate the noisy point with `t/T`.
- [ ] Run `python check_forward.py --part model`.
- [ ] Read `train.py`; explain the target and each tensor's shape.
- [ ] Use the five-update VS Code debug configuration. Set a breakpoint after
      `predicted_noise = model(xt, t)` and inspect x0, t, noise, xt and predicted_noise.
- [ ] Confirm that the model gets neither x0 nor the true sampled noise as inputs.

The provided training objective is mean squared error between sampled and
predicted noise. A zero predictor has expected elementwise MSE about one.
The optimal predictor is a conditional expectation, so loss need not reach zero.

### 4. Reverse diffusion: `diffusion.py`

- [ ] Implement `p_sample` with the DDPM reverse mean:

    mu = (x_t - beta_t / sqrt(1-alpha_bar_t) * predicted_noise) / sqrt(alpha_t)

- [ ] Use fixed posterior variance:

    posterior_variance_t = beta_t * (1-alpha_bar_(t-1)) / (1-alpha_bar_t)

- [ ] Add sqrt(posterior_variance_t) times fresh Gaussian noise when t>1.
      At t=1, return the mean without additional noise.
- [ ] Run `python check_forward.py --part reverse`.
- [ ] Run all checks: `python check_forward.py --part all`.

Use the complete timestep sequence initially. Skipping timesteps requires an
appropriate alternative sampler; simply skipping iterations is not DDIM.

### 5. Train and generate

From an allocated GPU shell (or use `--device cpu`):

```bash
python train.py --device cuda --steps 5000 --batch-size 256 --seed 42
python sample.py --device cuda --checkpoint outputs/run1/model.pt --num-samples 2000
```

The first settings are experimental defaults: T=1000, hidden width=128,
Adam with learning rate=1e-3, and float32. Sampling uses fresh Gaussian noise,
saved schedule settings, eval mode and no gradients. Reference data is drawn
only after generation, for plotting.

Outputs:
- `outputs/run1/model.pt`: weights, configuration and schedule, saved at the end.
- `outputs/run1/loss.csv`: loss logged every 100 updates.
- `outputs/run1/generated.png`: reverse snapshots and reference points.
- `outputs/run1/generated.pt`: generated points.

The checkpoints do not include optimizer state or provide training resume.
For these short learning runs, rerun training if interrupted.

### 6. Experiments and explanations

- [ ] Record predictions and observations in `notes.md`.
- [ ] Remove time conditioning, reduce T with unchanged beta endpoints, and
      remove reverse noise. Change one thing at a time.
- [ ] Compare cluster proportions, spread, overall mean and variance.
- [ ] Verify generation works from a saved checkpoint in a new process.

## Slurm template

Edit `train.slurm` for your GPU partition, account, module/container setup and
GPU resource naming. To request specifically an A100, follow your cluster's
GPU-type instructions. The generic `--gres=gpu:1` does not guarantee an A100.
Create the log directory before submission:

```bash
mkdir -p outputs
sbatch train.slurm
```

The template uses `.venv/bin/python` by default. To use another environment:

```bash
DM_PYTHON=/absolute/path/to/environment/bin/python sbatch train.slurm
```

For interactive GPU learning, use your site's allocation instructions. Within
the allocated shell, verify `nvidia-smi` and `torch.cuda.is_available()` before
training. See the [Slurm documentation](https://slurm.schedmd.com/sbatch.html).

## References

- [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion): focus on
  the forward process, objective and sampling. Leave the image U-Net for later.
- [DDPM paper](https://arxiv.org/abs/2006.11239): equations 4 and 11, Algorithms 1 and 2.
- [MIT course and Colab labs](https://diffusion.csail.mit.edu/2025/): a later route
  to score matching and flow matching; their conventions differ from discrete DDPM.

Suggested workflow: read an equation, close the reference, implement and test it,
then compare implementations and explain differences in `notes.md`.
