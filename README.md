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

- [*] Implement `sample_data`: choose uniformly among (-1,-1), (-1,1), (1,-1),
      (1,1), and add Gaussian noise with standard deviation 0.1.
- [*] Return float32 tensors of shape `[B, 2]` on the requested device.
- [*] Run `python check_forward.py --part data` and inspect `outputs/checks/data.png`.

### 2. Forward diffusion: `diffusion.py`

- [*] Implement `make_schedule` and `q_sample`.
- [*] Use mathematical indexing: arrays of length `T+1`, clean data at index 0,
      and training indices `1..T`.
- [*] Implement the following equations in PyTorch:

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

- [*] Implement `NoisePredictor.forward`: concatenate the noisy point with `t/T`.
- [*] Run `python check_forward.py --part model`.
- [*] Read `train.py`; explain the target and each tensor's shape.
- [*] Use the five-update VS Code debug configuration. Set a breakpoint after
      `predicted_noise = model(xt, t)` and inspect x0, t, noise, xt and predicted_noise.
- [*] Confirm that the model gets neither x0 nor the true sampled noise as inputs.

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

## SCAI Slurm job: one A100

`train.slurm` requests one node, one task, one A100 GPU, two CPU cores,
8 GB of host RAM and 30 minutes. It trains and then samples on the same allocation.
A startup check requires PyTorch >= 2.0, working CUDA and exactly one visible
A100 GPU, and rejects GPU names identifying MIG instances.

[SCAI's public specifications](https://scai.sorbonne-universite.fr/computing-power)
list NVIDIA A100-SXM4-40GB hardware, but do not publish Slurm partition, account,
QoS or GPU resource names. The script assumes `--gres=gpu:a100:1`; verify the
actual GPU type and partition on the cluster before your first submission:

```bash
sinfo -N -o '%P %N %G %f'
```

Use the GPU type shown in the GRES column. No partition, account or QoS is
hard-coded. If the site's defaults support your A100 request, submit from DM_ex:

```bash
mkdir -p outputs
sbatch train.slurm
```

The script requests email on job completion or failure (`END,FAIL`), after
both training and sampling finish. A blank recipient placeholder is provided
in `train.slurm` as `##SBATCH --mail-user=`. Fill in your email and remove one
leading `#` to activate it, or specify your recipient when submitting:

```bash
sbatch --mail-user=YOUR_EMAIL train.slurm
```

Replace `YOUR_EMAIL` with your address. Without `--mail-user`, Slurm defaults
to the submitting username using the cluster's configured mail domain.
Delivery requires the cluster's Slurm mail service to be configured. These
options apply to newly submitted jobs.

Otherwise provide your confirmed partition and GPU type at submission; the
following capitalized values are placeholders, not SCAI configuration:

```bash
sbatch --partition=YOUR_A100_PARTITION --gres=gpu:YOUR_A100_TYPE:1 train.slurm
```

Add `--account=YOUR_ACCOUNT` and/or `--qos=YOUR_QOS` if your access requires them.
For untyped GPU resources, use `--gres=gpu:1` together with a confirmed A100-only
partition, node selection or advertised constraint. The startup check detects a
wrong GPU after allocation; it does not replace scheduler resource selection.

Slurm opens `outputs/slurm-JOB_ID.log` before the script starts, so create
`outputs` before calling `sbatch`. Training and sampling results go to
`outputs/run1`, which is ignored by Git. Repeated jobs overwrite that run's results.

The script uses `.venv/bin/python` by default. To use another CUDA-enabled
PyTorch environment:

```bash
DM_PYTHON=/absolute/path/to/environment/bin/python sbatch train.slurm
```

Use SCAI's documented module or container setup if your account requires it.
The script preserves Slurm's `CUDA_VISIBLE_DEVICES`; do not select a physical
GPU number yourself. For interactive learning, follow the site's allocation
instructions. See the [Slurm submission documentation](https://slurm.schedmd.com/sbatch.html).

## References

- [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion): focus on
  the forward process, objective and sampling. Leave the image U-Net for later.
- [DDPM paper](https://arxiv.org/abs/2006.11239): equations 4 and 11, Algorithms 1 and 2.
- [MIT course and Colab labs](https://diffusion.csail.mit.edu/2025/): a later route
  to score matching and flow matching; their conventions differ from discrete DDPM.

Suggested workflow: read an equation, close the reference, implement and test it,
then compare implementations and explain differences in `notes.md`.
