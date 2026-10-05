"""Training harness. Complete data.py, diffusion.py and model.py first."""
import argparse
from pathlib import Path

import torch

from data import sample_data
from diffusion import make_schedule, q_sample
from model import NoisePredictor
from utils import get_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/run1"))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.timesteps, args.hidden) < 1 or args.lr <= 0:
        parser.error("Steps, batch size, timesteps, hidden width, and learning rate must be positive")

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    schedule = make_schedule(args.timesteps, device)
    model = NoisePredictor(args.timesteps, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    model.train()

    with (args.output / "loss.csv").open("w") as log:
        log.write("step,loss\n")
        for step in range(1, args.steps + 1):
            x0 = sample_data(args.batch_size, device)
            t = torch.randint(1, args.timesteps + 1, (args.batch_size,), device=device)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, schedule)
            predicted_noise = model(xt, t)
            loss = (predicted_noise - noise).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == args.steps:
                value = loss.item()
                print(f"step={step} loss={value:.5f}", flush=True)
                log.write(f"{step},{value}\n")
                log.flush()

    config = vars(args).copy()
    config["output"] = str(config["output"])
    checkpoint = {
        "model": model.state_dict(), "config": config,
        "schedule": {k: v.cpu() for k, v in schedule.items()},
        "torch_version": str(torch.__version__),
    }
    torch.save(checkpoint, args.output / "model.pt")
    print(f"Saved {args.output / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
