import torch

def smooth_delta(phi: torch.Tensor, eps: float):
    return (eps / (torch.pi * (eps**2 + phi**2)))

def area_loss(pred_sdf: torch.Tensor, query_points: torch.Tensor, eps: float = 0.1):
    grads = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=query_points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (smooth_delta(pred_sdf, eps) * grads.norm(dim=-1)).mean()