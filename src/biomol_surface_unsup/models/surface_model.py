import torch
import torch.nn as nn

from biomol_surface_unsup.features.local_features import LocalFeatureBuilder
from biomol_surface_unsup.features.global_features import GlobalFeatureEncoder
from biomol_surface_unsup.models.encoders.local_deepsets import LocalDeepSetsEncoder
from biomol_surface_unsup.models.decoders.sdf_decoder import SDFDecoder
from biomol_surface_unsup.models.fusion import concat_fusion


class SurfaceModel(nn.Module):
    def __init__(self, num_atom_types: int, cutoff: float = 8.0, max_neighbors: int = 64):
        super().__init__()
        self.local_builder = LocalFeatureBuilder(
            num_atom_types=num_atom_types,
            atom_embed_dim=16,
            rbf_dim=16,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        self.local_encoder = LocalDeepSetsEncoder(
            in_dim=self.local_builder.feature_dim,
            hidden_dim=128,
            out_dim=128,
        )
        self.global_encoder = GlobalFeatureEncoder(
            num_atom_types=num_atom_types,
            atom_embed_dim=16,
            hidden_dim=128,
            out_dim=128,
        )
        self.decoder = SDFDecoder(in_dim=256, hidden_dim=128)

    def forward(self, coords, atom_types, radii, query_points, atom_mask=None, query_mask=None):
        """Forward pass for batched or single-sample toy inputs.

        Batched path:
        - coords: [B, N, 3]
        - atom_types: [B, N]
        - radii: [B, N]
        - atom_mask: [B, N]
        - query_points: [B, Q, 3]
        - query_mask: [B, Q]
        - sdf: [B, Q]

        Single-sample compatibility:
        - 2D/1D inputs are promoted to batch size 1 and squeezed back on return.
        """
        squeeze_batch = coords.ndim == 2
        local = self.local_builder(
            coords,
            atom_types,
            radii,
            query_points,
            atom_mask=atom_mask,
            query_mask=query_mask,
        )
        z_local = self.local_encoder(local["features"], local["mask"])
        z_global = self.global_encoder(coords, atom_types, radii, atom_mask=atom_mask)
        if z_local.ndim == 2:
            z_global_expanded = z_global.unsqueeze(0).expand(query_points.shape[0], -1)
        else:
            z_global_expanded = z_global.unsqueeze(1).expand(-1, z_local.shape[1], -1)
        z = concat_fusion(z_local, z_global_expanded)
        sdf = self.decoder(z)
        if query_mask is not None:
            sdf = sdf * query_mask.to(sdf.dtype)
        return {"sdf": sdf, "z_local": z_local, "z_global": z_global_expanded, **local}
