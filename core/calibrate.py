import torch
import torch.nn as nn

from core.losses import gaussian_nll


class SigmaCalibrator(nn.Module):
    
    #sigma_cal = exp(log_scale) * sigma_pred   (per target)
    #log_sigma_cal = log_sigma_pred + log_scale
    
    def __init__(self, d_out: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(d_out))

    def forward(self, log_sigma_pred: torch.Tensor) -> torch.Tensor:
        return log_sigma_pred + self.log_scale[None, :]


@torch.no_grad()
def predict_normalized(model, loader, normalizer, device: str):
    model.eval()
    mus, logs, ys = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_n = normalizer.encode(y)
        mu_n, log_sigma_n = model(x)
        mus.append(mu_n.cpu())
        logs.append(log_sigma_n.cpu())
        ys.append(y_n.cpu())
    return torch.cat(mus, dim=0), torch.cat(logs, dim=0), torch.cat(ys, dim=0)


def fit_sigma_calibrator(mu_n, log_sigma_n, y_n, lr: float = 0.05, steps: int = 500):
    d_out = y_n.shape[1]
    calib = SigmaCalibrator(d_out)
    opt = torch.optim.Adam([calib.log_scale], lr=lr)

    mu_n = mu_n.detach()
    log_sigma_n = log_sigma_n.detach()
    y_n = y_n.detach()

    for _ in range(steps):
        opt.zero_grad()
        log_sigma_cal = calib(log_sigma_n)
        loss = gaussian_nll(y_n, mu_n, log_sigma_cal)
        loss.backward()
        opt.step()

    return calib