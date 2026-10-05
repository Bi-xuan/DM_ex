"""Milestone checks: select data, forward, model, reverse, or all."""
import argparse
from pathlib import Path

import torch

from data import sample_data
from diffusion import make_schedule, p_sample, q_sample
from model import NoisePredictor
from utils import get_device, scatter_panels, seed_everything


def check_data(device, output):
    points = sample_data(20000, device)
    assert points.shape == (20000, 2), "Return one [x, y] pair per point"
    assert points.dtype == torch.float32 and points.device == device
    assert torch.isfinite(points).all()
    centers = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], device=device)
    labels = torch.cdist(points, centers).argmin(1)
    frequencies = torch.bincount(labels, minlength=4).float() / len(points)
    assert ((frequencies - 0.25).abs() < 0.025).all(), "Check uniform center sampling"
    residuals = points - centers[labels]
    torch.testing.assert_close(residuals.mean(0), torch.zeros(2, device=device), atol=0.005, rtol=0)
    torch.testing.assert_close(residuals.std(0), torch.full((2,), 0.1, device=device), atol=0.005, rtol=0)
    scatter_panels([("training distribution", points[:2000])], output / "data.png")
    print("Data checks passed. Cluster proportions:", frequencies.cpu().tolist())


def check_forward(device, output):
    T = 1000
    schedule = make_schedule(T, device)
    for key in ("beta", "alpha", "alpha_bar", "posterior_variance"):
        value = schedule[key]
        assert value.shape == (T + 1,) and value.device == device, key
        assert torch.isfinite(value).all(), key
    ab = schedule["alpha_bar"]
    assert ab[0].item() == 1 and (ab[1:] < ab[:-1]).all()
    assert ab[-1].item() < 0.001, "Terminal corruption should be nearly Gaussian"
    assert schedule["posterior_variance"][1].item() == 0
    assert (schedule["posterior_variance"] >= 0).all()

    x0 = sample_data(2000, device)
    noise = torch.randn_like(x0)
    zero = torch.zeros(len(x0), dtype=torch.long, device=device)
    torch.testing.assert_close(q_sample(x0, zero, noise, schedule), x0)
    times = torch.randint(1, T + 1, (len(x0),), device=device)
    xt = q_sample(x0, times, noise, schedule)
    assert xt.shape == x0.shape
    a = ab[times].unsqueeze(1)
    recovered = (xt - (1 - a).sqrt() * noise) / a.sqrt()
    torch.testing.assert_close(recovered, x0, atol=1e-4, rtol=1e-4)

    # Independent statistical check: direct versus repeated one-step corruption.
    target = 200
    fixed = torch.tensor([[1., -1.]], device=device).repeat(20000, 1)
    direct = q_sample(fixed, torch.full((len(fixed),), target, device=device, dtype=torch.long),
                      torch.randn_like(fixed), schedule)
    chained = fixed.clone()
    for t in range(1, target + 1):
        chained = schedule["alpha"][t].sqrt() * chained + schedule["beta"][t].sqrt() * torch.randn_like(chained)
    expected_mean = ab[target].sqrt() * fixed[0]
    expected_var = (1 - ab[target]).expand(2).clone()
    for points in (direct, chained):
        torch.testing.assert_close(points.mean(0), expected_mean, atol=0.025, rtol=0)
        torch.testing.assert_close(points.var(0), expected_var, atol=0.025, rtol=0)

    panels = [("clean t=0", x0)]
    for t in (100, 300, 600, 1000):
        batch_t = torch.full((len(x0),), t, dtype=torch.long, device=device)
        panels.append((f"t={t}", q_sample(x0, batch_t, torch.randn_like(x0), schedule)))
    scatter_panels(panels, output / "forward.png")
    print("Forward checks passed, including batched timesteps and direct/chained statistics.")


def check_model(device, output):
    model = NoisePredictor().to(device)
    xt = torch.randn(16, 2, device=device)
    times = torch.randint(1, 1001, (16,), device=device)
    prediction = model(xt, times)
    assert prediction.shape == xt.shape and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("Model shape and gradient checks passed.")


def check_reverse(device, output):
    schedule = make_schedule(1000, device)

    class ZeroNoise(torch.nn.Module):
        def forward(self, xt, t):
            assert t.shape == (len(xt),) and t.dtype == torch.long
            return torch.zeros_like(xt)

    model = ZeroNoise()
    points = torch.randn(16, 2, device=device)
    first = p_sample(model, points, 1, schedule)
    second = p_sample(model, points, 1, schedule)
    assert first.shape == points.shape
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first, points / schedule["alpha"][1].sqrt())
    # At t>1, randomness must have the variance specified by this sampler.
    fixed = torch.zeros(20000, 2, device=device)
    drawn = p_sample(model, fixed, 500, schedule)
    variance = schedule["posterior_variance"][500].expand(2).clone()
    torch.testing.assert_close(drawn.mean(0), torch.zeros(2, device=device), atol=0.005, rtol=0)
    torch.testing.assert_close(drawn.var(0), variance, atol=0.001, rtol=0.05)
    print("Reverse checks passed: deterministic final step and stochastic earlier step.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("data", "forward", "model", "reverse", "all"), default="forward")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/checks"))
    args = parser.parse_args()
    seed_everything(42)
    device = get_device(args.device)
    checks = {"data": check_data, "forward": check_forward, "model": check_model, "reverse": check_reverse}
    for name, check in checks.items():
        if args.part in (name, "all"):
            check(device, args.output)


if __name__ == "__main__":
    main()
