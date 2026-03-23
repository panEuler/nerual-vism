import torch
from biomol_surface_unsup.geometry.sdf_ops import atomic_union_field

def weak_prior_loss(coords, radii, query_points, pred_sdf):
    target = atomic_union_field(coords, radii, query_points)
    return (pred_sdf - target).abs().mean()