import torch

def containment_loss(center_sdf: torch.Tensor, margin: float = 0.5):
    return torch.relu(center_sdf + margin).pow(2).mean()