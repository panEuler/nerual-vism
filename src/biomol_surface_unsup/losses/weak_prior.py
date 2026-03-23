from __future__ import annotations

import torch


def _batched_atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    pairwise = torch.cdist(query_points, coords) - radii.unsqueeze(-2)
    return -torch.logsumexp(-10.0 * pairwise, dim=-1) / 10.0


def weak_prior_loss(
    coords: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Toy weak prior against the atomic-union proxy.

    Batched shapes:
    - coords: [B, N, 3]
    - radii: [B, N]
    - query_points: [B, Q, 3]
    - pred_sdf: [B, Q]
    - mask: [B, Q] or None
    - atom_mask: [B, N] or None
    """
    squeeze_batch = coords.ndim == 2
    if squeeze_batch:
        coords = coords.unsqueeze(0)
        radii = radii.unsqueeze(0)
        query_points = query_points.unsqueeze(0)
        pred_sdf = pred_sdf.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)
        atom_mask = None if atom_mask is None else atom_mask.unsqueeze(0)

    if atom_mask is None:
        atom_mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    safe_radii = radii.masked_fill(~atom_mask, 0.0)
    safe_coords = coords.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
    target = _batched_atomic_union_field(safe_coords, safe_radii, query_points).detach()

    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        return (pred_sdf[mask] - target[mask]).abs().mean()
    return (pred_sdf - target).abs().mean()
