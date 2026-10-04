import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core.config import cfg
from core.data import TransitNPZDataset, make_splits
from core.losses import gaussian_nll, beta_nll

from core.normalizer import TargetNormalizer
from core.calibrate import predict_normalized, fit_sigma_calibrator
from core.metrics import eval_metrics, pretty_print_metrics


NPZ_PATH        = cfg["paths"]["train_data"]
SAVE_MODEL_PATH = cfg["paths"]["model"]
SAVE_AUX_PATH   = cfg["paths"]["aux"]

SEED = 42

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)


BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 300
NUM_WORKERS = 0
PIN_MEMORY = False

TRAIN_FRAC = 0.75
VAL_FRAC = 0.10
TEST_FRAC = 0.15

USE_TIME_CHANNEL = True
CENTER_FLUX = True

LOGSIGMA_MIN = -8.0
LOGSIGMA_MAX = 4.0

BETA_PER_DIM = torch.tensor([0.0, 0.5, 0.0], device=DEVICE)   # duration only 


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    print("Device:", DEVICE)

    ds = TransitNPZDataset(NPZ_PATH, use_time=USE_TIME_CHANNEL, center_flux=CENTER_FLUX)
    n = len(ds)
    train_idx, val_idx, test_idx = make_splits(n, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, SEED)

    train_set = Subset(ds, train_idx.tolist())
    val_set = Subset(ds, val_idx.tolist())
    test_set = Subset(ds, test_idx.tolist())

    # default log-mask: log only on duration (col 1). depth and dtcorr linear.
    normalizer = TargetNormalizer.fit_from_subset(ds, train_set, log_mask=cfg["log_mask"])
    print("log_mask (log columns):", normalizer.log_mask)
    print("Target mean [depth, log(dur), dtcorr]:", normalizer.mean)
    print("Target std  [depth, log(dur), dtcorr]:", normalizer.std)

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate
    )

    in_ch = 2 if USE_TIME_CHANNEL else 1
    #model = cfg["model_class"](
    #    in_ch=in_ch, d_out=3, log_sigma_min=LOGSIGMA_MIN, log_sigma_max=LOGSIGMA_MAX).to(DEVICE)

    model = cfg["model_class"](
            in_ch=in_ch, d_out=3, log_sigma_min=LOGSIGMA_MIN, log_sigma_max=LOGSIGMA_MAX,
            dt_mean=float(normalizer.mean[2]), dt_std=float(normalizer.std[2])).to(DEVICE)

    with torch.no_grad():
        x0, _ = ds[0]
        L = x0.shape[-1]  # actual window length
        h_dbg = model.feat(torch.randn(2, in_ch, L).to(DEVICE))
        print("feature length:", h_dbg.shape[-1])  
    # --------------------------------------------------------

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    best_val_nll = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_n = normalizer.encode(y)

            mu_n, log_sigma_n = model(x)
            loss = gaussian_nll(y_n, mu_n, log_sigma_n)

            #mu_n, log_sigma_n = model(x)
            #loss = beta_nll(y_n, mu_n, log_sigma_n, beta_per_dim=BETA_PER_DIM)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            bs = y.shape[0]
            running += loss.item() * bs
            seen += bs

        train_nll = running / seen
        val_m = eval_metrics(model, val_loader, normalizer, DEVICE, calib=None)
        val_nll = val_m["nll"]
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} | train NLL={train_nll:.4f} | val NLL={val_nll:.4f} | LR={current_lr:.2e}")

        scheduler.step(val_nll)

        if epoch >= 10 and val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pretty_print_metrics("VAL (uncalibrated)", eval_metrics(model, val_loader, normalizer, DEVICE, calib=None))
    pretty_print_metrics("TEST (uncalibrated)", eval_metrics(model, test_loader, normalizer, DEVICE, calib=None))

    mu_n_val, log_sigma_n_val, y_n_val = predict_normalized(model, val_loader, normalizer, DEVICE)
    calib = fit_sigma_calibrator(mu_n_val, log_sigma_n_val, y_n_val, lr=0.05, steps=500).to(DEVICE)

    print("\nCalibrator log_scale:", calib.log_scale.detach().cpu().numpy())
    print("Calibrator scale factors:", torch.exp(calib.log_scale).detach().cpu().numpy())

    pretty_print_metrics("VAL (calibrated)", eval_metrics(model, val_loader, normalizer, DEVICE, calib=calib))
    pretty_print_metrics("TEST (calibrated)", eval_metrics(model, test_loader, normalizer, DEVICE, calib=calib))

    # save
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": "cnn",
            "use_time_channel": USE_TIME_CHANNEL,
            "center_flux": CENTER_FLUX,
            "log_sigma_min": LOGSIGMA_MIN,
            "log_sigma_max": LOGSIGMA_MAX,
        },
        SAVE_MODEL_PATH,
    )
    np.savez_compressed(
        SAVE_AUX_PATH,
        y_mean=normalizer.mean,
        y_std=normalizer.std,
        y_log_mask=normalizer.log_mask,          # required for inference
        calib_log_scale=calib.log_scale.detach().cpu().numpy(),
    )

    print("Saved model:", SAVE_MODEL_PATH)
    print("Saved aux  :", SAVE_AUX_PATH)


if __name__ == "__main__":
    main()