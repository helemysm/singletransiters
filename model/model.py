
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int, p: int,
                 dropout: float, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



#Predicts dtcorr as the expected position over the time bins.
class SoftArgmaxDtHead(nn.Module):
 
    def __init__(self, in_ch: int, window_hours: float = 99.5):
        super().__init__()
        self.window_hours = window_hours
        self.score = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, h: torch.Tensor):
        B, C, T = h.shape
        half = self.window_hours / 2.0
        pos = torch.linspace(-half, half, T, device=h.device, dtype=h.dtype)  # (T,)
        score = self.score(h).squeeze(1)
        w = torch.softmax(score, dim=-1) 
        mu = (w * pos).sum(dim=-1, keepdim=True) # (B, 1) hours
        var = (w * (pos - mu) ** 2).sum(dim=-1, keepdim=True)  
        sigma = torch.sqrt(var + 1e-6) # (B, 1) hours
        return mu, sigma



class SmallConvUncRegressor(nn.Module):
    
    def __init__(self, in_ch: int, d_out: int = 3,
                 log_sigma_min: float = -8.0, log_sigma_max: float = 4.0,
                 window_hours: float = 99.5,
                 dt_mean: float = 0.0, dt_std: float = 20.0):
        super().__init__()
        self.d_out = d_out
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.feat = nn.Sequential(
            ConvBlock(in_ch, 32,  k=7, stride=1, p=3, dropout=0.05),
            ConvBlock(32,   64,  k=5, stride=2, p=2, dropout=0.05),
            ConvBlock(64,  128,  k=5, stride=1, p=4, dropout=0.05, dilation=2),
            ConvBlock(128, 128,  k=3, stride=2, p=2, dropout=0.05, dilation=2),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        C = 128

        # dtcorr normalization buffers (saved/restored with the state_dict)
        self.register_buffer("dt_mean", torch.tensor(float(dt_mean)))
        self.register_buffer("dt_std",  torch.tensor(float(dt_std)))

        # depth: avg only (128)
        self.mu_head_depth = nn.Sequential(
            nn.Linear(C, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))
        self.sigma_head_depth = nn.Sequential(
            nn.Linear(C, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))

        # duration: avg + std (256)
        self.mu_head_dur = nn.Sequential(
            nn.Linear(2 * C, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))
        self.sigma_head_dur = nn.Sequential(
            nn.Linear(2 * C, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))

        # dtcorr: positional soft-argmax
        self.dt_head = SoftArgmaxDtHead(C, window_hours=window_hours)

        # dtcorr sigma: from h_avg + positional spread of the soft-argmax
        self.sigma_head_dt = nn.Sequential(
            nn.Linear(C + 1, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1))


    def forward(self, x: torch.Tensor):
        h = self.feat(x)                                

        h_avg = self.avg_pool(h).squeeze(-1)            
        h_std = h.std(dim=-1)                          
        h_cat_dur = torch.cat([h_avg, h_std], dim=1)   

        mu_depth = self.mu_head_depth(h_avg)           
        ls_depth = self.sigma_head_depth(h_avg)        

        mu_dur = self.mu_head_dur(h_cat_dur)           
        ls_dur = self.sigma_head_dur(h_cat_dur) 

        # dtcorr in physical hours -> normalized space
        mu_dt_phys, sigma_dt_phys = self.dt_head(h)
        mu_dt = (mu_dt_phys - self.dt_mean) / self.dt_std
        sigma_feat = torch.cat([h_avg, sigma_dt_phys], dim=1)
        ls_dt = self.sigma_head_dt(sigma_feat) # sigma from h_avg + positional spread

        mu = torch.cat([mu_depth, mu_dur, mu_dt], dim=1)       
        log_sigma = torch.cat([ls_depth, ls_dur, ls_dt], dim=1)
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma
