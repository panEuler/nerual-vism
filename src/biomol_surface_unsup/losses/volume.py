import torch

def smooth_heaviside(x: torch.Tensor, eps: float):
    return 0.5 * (1.0 + (2.0 / torch.pi) * torch.atan(x / eps))

def volume_loss(pred_sdf: torch.Tensor, target_volume_fraction: float = 0.5, eps: float = 0.1):
    inside = smooth_heaviside(-pred_sdf, eps).mean()
    return (inside - target_volume_fraction) ** 2