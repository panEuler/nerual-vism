import torch

def sphere_sdf(query_points: torch.Tensor, center: torch.Tensor, radius: torch.Tensor):
    return (query_points - center).norm(dim=-1) - radius

def smooth_min(x: torch.Tensor, dim: int = -1, temperature: float = 10.0):
    return -torch.logsumexp(-temperature * x, dim=dim) / temperature

def atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor):
    d = torch.cdist(query_points, coords) - radii.unsqueeze(0)
    return smooth_min(d, dim=-1)