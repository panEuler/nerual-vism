from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None


def sample_query_points(coords: Any, num_query_points: int, padding: float):
    """Create deterministic toy query points with shape [Q, 3]."""
    del padding
    if torch is not None and isinstance(coords, torch.Tensor):
        base = coords.mean(dim=0, keepdim=True)
        offsets = torch.linspace(-1.0, 1.0, steps=num_query_points, dtype=coords.dtype, device=coords.device)
        query_points = torch.stack(
            [offsets, torch.zeros_like(offsets), -offsets],
            dim=1,
        )
        return base + query_points  # [Q, 3]

    base = [0.0, 0.0, 0.0]
    if coords:
        count = float(len(coords))
        base = [
            sum(point[0] for point in coords) / count,
            sum(point[1] for point in coords) / count,
            sum(point[2] for point in coords) / count,
        ]

    if num_query_points <= 1:
        offsets = [0.0]
    else:
        step = 2.0 / float(num_query_points - 1)
        offsets = [-1.0 + step * index for index in range(num_query_points)]

    return [[base[0] + offset, base[1], base[2] - offset] for offset in offsets]


def sample_surface_band_points(coords: Any, num_points: int):
    """Return near-center toy points with shape [Q, 3]."""
    return sample_query_points(coords=coords, num_query_points=num_points, padding=0.0)
