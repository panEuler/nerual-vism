from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - environment fallback
    torch = None
    nn = None
    DataLoader = None

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset


def _ensure_local_feature_builder() -> None:
    """Install a minimal toy LocalFeatureBuilder expected by SurfaceModel."""
    if torch is None or nn is None:
        return

    import biomol_surface_unsup.features.local_features as local_features_module

    if hasattr(local_features_module, "LocalFeatureBuilder"):
        return

    class LocalFeatureBuilder(nn.Module):
        def __init__(
            self,
            num_atom_types: int,
            atom_embed_dim: int,
            rbf_dim: int,
            cutoff: float = 8.0,
            max_neighbors: int = 64,
        ) -> None:
            super().__init__()
            self.num_atom_types = num_atom_types
            self.atom_embed_dim = atom_embed_dim
            self.rbf_dim = rbf_dim
            self.cutoff = cutoff
            self.max_neighbors = max_neighbors
            self.feature_dim = 3 + 1 + 1

        def forward(
            self,
            coords: torch.Tensor,
            atom_types: torch.Tensor,
            radii: torch.Tensor,
            query_points: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del atom_types

            num_queries = query_points.shape[0]
            num_atoms = coords.shape[0]
            num_neighbors = min(num_atoms, self.max_neighbors)

            deltas = query_points[:, None, :] - coords[None, :, :]  # [Q, N, 3]
            distances = deltas.norm(dim=-1)  # [Q, N]
            sorted_distances, sorted_indices = torch.sort(distances, dim=1)
            neighbor_indices = sorted_indices[:, :num_neighbors]  # [Q, K]
            neighbor_distances = sorted_distances[:, :num_neighbors]  # [Q, K]

            gather_index = neighbor_indices.unsqueeze(-1).expand(-1, -1, 3)
            neighbor_deltas = torch.gather(deltas, dim=1, index=gather_index)  # [Q, K, 3]
            neighbor_radii = torch.gather(radii.unsqueeze(0).expand(num_queries, -1), dim=1, index=neighbor_indices)  # [Q, K]

            features = torch.cat(
                [
                    neighbor_deltas,
                    neighbor_distances.unsqueeze(-1),
                    neighbor_radii.unsqueeze(-1),
                ],
                dim=-1,
            )  # [Q, K, 5]
            mask = torch.ones(
                (num_queries, num_neighbors),
                dtype=query_points.dtype,
                device=query_points.device,
            )  # [Q, K]
            return {"features": features, "mask": mask}

    local_features_module.LocalFeatureBuilder = LocalFeatureBuilder


def _build_loss_fn(cfg):
    try:
        from biomol_surface_unsup.losses.loss_builder import build_loss_fn

        return build_loss_fn(cfg)
    except Exception:
        from biomol_surface_unsup.geometry.sdf_ops import atomic_union_field
        from biomol_surface_unsup.losses.area import area_loss
        from biomol_surface_unsup.losses.volume import volume_loss

        def loss_fn(batch, model_out):
            coords = batch["coords"]  # [N, 3]
            radii = batch["radii"]  # [N]
            query_points = batch["query_points"]  # [Q, 3]
            pred_sdf = model_out["sdf"]  # [Q]
            target_sdf = atomic_union_field(coords, radii, query_points)  # [Q]
            prior = (pred_sdf - target_sdf).abs().mean()
            eikonal = torch.zeros((), device=pred_sdf.device, dtype=pred_sdf.dtype)
            total = area_loss(pred_sdf, query_points) + 0.5 * volume_loss(pred_sdf) + 0.5 * prior + 0.1 * eikonal
            return {
                "area": area_loss(pred_sdf, query_points),
                "volume": volume_loss(pred_sdf),
                "prior": prior,
                "eikonal": eikonal,
                "total": total,
            }

        return loss_fn


class Trainer:
    def __init__(self, cfg):
        if torch is None or DataLoader is None:
            raise RuntimeError("torch is required to run Trainer in this environment")

        _ensure_local_feature_builder()
        from biomol_surface_unsup.models.surface_model import SurfaceModel
        from biomol_surface_unsup.training.optimizer import build_optimizer
        from biomol_surface_unsup.training.train_step import train_step

        self._train_step = train_step
        self.cfg = cfg
        requested_device = str(cfg["train"].get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        self.train_dataset = MoleculeDataset(
            root=data_cfg.get("root", "data/processed/toy"),
            split=data_cfg.get("train_split", "train"),
            num_samples=1,
            num_atoms=4,
            num_query_points=int(data_cfg.get("num_query_points", 32)),
            bbox_padding=float(data_cfg.get("bbox_padding", 2.0)),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(train_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
        )

        self.model = SurfaceModel(num_atom_types=16).to(self.device)
        self.loss_fn = _build_loss_fn(cfg)
        self.optimizer = build_optimizer(
            self.model,
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )

    def train(self):
        num_epochs = int(self.cfg["train"].get("epochs", 1))
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.train_loader):
                metrics = self._train_step(self.model, batch, self.loss_fn, self.optimizer, self.device)
                print(f"epoch={epoch} step={step} metrics={metrics}")

    def evaluate(self):
        print("TODO")
