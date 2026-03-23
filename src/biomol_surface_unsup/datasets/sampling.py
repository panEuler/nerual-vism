from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None


def sample_query_points(coords: Any, num_query_points: int, padding: float):
    """Create deterministic toy query points with shape [Q, 3]."""
    if torch is not None and isinstance(coords, torch.Tensor):
        base = coords.mean(dim=0, keepdim=True)  # [1, 3]
        span = max(float(padding), 1.0)
        offsets = torch.linspace(-span, span, steps=max(num_query_points, 1), dtype=coords.dtype, device=coords.device)  # [Q]
        query_points = torch.stack(
            [offsets, torch.zeros_like(offsets), -0.5 * offsets],
            dim=1,
        )  # [Q, 3]
        return base + query_points  # [Q, 3]

    base = [0.0, 0.0, 0.0]
    if coords:
        count = float(len(coords))
        base = [
            sum(point[0] for point in coords) / count,
            sum(point[1] for point in coords) / count,
            sum(point[2] for point in coords) / count,
        ]

    span = max(float(padding), 1.0)
    if num_query_points <= 1:
        offsets = [0.0]
    else:
        step = (2.0 * span) / float(num_query_points - 1)
        offsets = [-span + step * index for index in range(num_query_points)]

    return [[base[0] + offset, base[1], base[2] - 0.5 * offset] for offset in offsets]


def sample_surface_band_points(coords: Any, num_points: int):
    """Return near-center toy points with shape [Q, 3]."""
    return sample_query_points(coords=coords, num_query_points=num_points, padding=1.0)
