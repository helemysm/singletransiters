from pathlib import Path
import numpy as np
import yaml
import model.model as model_module

with open(Path(__file__).parent / "config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["model_class"] = getattr(model_module, cfg["model_class"])

_norm = cfg.get("normalizer", {}) or {}
cfg["log_duration"] = bool(_norm.get("log_duration", True))
cfg["log_mask"] = np.array([False, cfg["log_duration"], False], dtype=bool)


def resolve_log_mask(aux=None):
   
    if aux is not None and "y_log_mask" in getattr(aux, "files", []):
        return np.asarray(aux["y_log_mask"], dtype=bool)
    return cfg["log_mask"]