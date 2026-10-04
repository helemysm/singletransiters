import numpy as np
import torch

from core.losses import gaussian_nll


@torch.no_grad()
def eval_metrics(model, loader, normalizer, device: str, calib=None):
    model.eval()

    mae_sum = torch.zeros(3)
    rmse_sum = torch.zeros(3)
    nll_sum = 0.0
    n = 0

    cov68 = torch.zeros(3)
    cov95 = torch.zeros(3)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_n = normalizer.encode(y)

        mu_n, log_sigma_n = model(x)
        if calib is not None:
            log_sigma_n = calib(log_sigma_n)

        # NLL in normalized space
        nll = gaussian_nll(y_n, mu_n, log_sigma_n).item()

        mu, sigma = normalizer.decode_mu_sigma(mu_n, log_sigma_n)
        err = (mu - y).abs()
        mae_sum += err.sum(dim=0).cpu()
        rmse_sum += ((mu - y) ** 2).sum(dim=0).cpu()

        z = (y - mu).abs() / (sigma + 1e-12)
        cov68 += (z <= 1.0).float().sum(dim=0).cpu()
        cov95 += (z <= 1.96).float().sum(dim=0).cpu()

        bs = y.shape[0]
        n += bs
        nll_sum += nll * bs

    mae = mae_sum / n
    rmse = torch.sqrt(rmse_sum / n)
    cov68 = cov68 / n
    cov95 = cov95 / n

    return {
        "N": n,
        "nll": nll_sum / n,
        "mae": mae.numpy(),
        "rmse": rmse.numpy(),
        "cov68": cov68.numpy(),
        "cov95": cov95.numpy(),
    }


def pretty_print_metrics(tag: str, m: dict):
    names = ["depth_ppm", "duration_h", "dtcorr_h"]
    print(f"\n== {tag} ==")
    print(f"N = {m['N']} | NLL = {m['nll']:.4f}")
    for j, name in enumerate(names):
        print(
            f"{name:>10s} | MAE={m['mae'][j]:9.3f} | RMSE={m['rmse'][j]:9.3f} | "
            f"Cov68={m['cov68'][j]:.3f} | Cov95={m['cov95'][j]:.3f}"
        )