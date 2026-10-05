"""Generate from fresh noise using a saved model, in a separate process."""
import argparse
from pathlib import Path

import torch

from data import sample_data
from diffusion import p_sample
from model import NoisePredictor
from utils import get_device, scatter_panels, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/run1/model.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("outputs/run1/generated.png"))
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("num-samples must be positive")
    seed_everything(args.seed)
    device = get_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    T = config["timesteps"]
    model = NoisePredictor(T, config["hidden"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    schedule = {k: v.to(device) for k, v in saved["schedule"].items()}
    xt = torch.randn(args.num_samples, 2, device=device)
    snapshots = [(f"initial t={T}", xt.cpu().clone())]
    with torch.no_grad():
        for t in range(T, 0, -1):
            xt = p_sample(model, xt, t, schedule)
            if t - 1 in {T // 2, T // 4}:
                snapshots.append((f"t={t - 1}", xt.cpu().clone()))
    if not torch.isfinite(xt).all():
        raise RuntimeError("Generated samples contain non-finite values")
    snapshots.append(("generated t=0", xt.cpu()))
    # Real points are drawn only AFTER generation, solely for comparison.
    snapshots.append(("reference data", sample_data(args.num_samples, device).cpu()))
    scatter_panels(snapshots, args.output)
    torch.save(xt.cpu(), args.output.with_suffix(".pt"))
    print(f"Saved {args.output}")
    print(f"Generated mean: {xt.mean(0).cpu().tolist()}")
    print(f"Generated variance: {xt.var(0).cpu().tolist()}")


if __name__ == "__main__":
    main()
