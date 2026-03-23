from __future__ import annotations

from typing import Any

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback when torch is unavailable
    class Dataset:  # type: ignore[override]
        pass

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None

from biomol_surface_unsup.datasets.sampling import sample_query_points


class MoleculeDataset(Dataset):
    """Toy dataset that returns a single fake molecule sample.

    Shapes:
    - coords: [N, 3]
    - atom_types: [N]
    - radii: [N]
    - query_points: [Q, 3]
    """

    def __init__(
        self,
        root: str = "data/processed/toy",
        split: str = "train",
        num_samples: int = 1,
        num_atoms: int = 4,
        num_query_points: int = 8,
        bbox_padding: float = 2.0,
    ) -> None:
        self.root = root
        self.split = split
        self.num_samples = num_samples
        self.num_atoms = num_atoms
        self.num_query_points = num_query_points
        self.bbox_padding = bbox_padding
        self.items = list(range(num_samples))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        coords_data = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
        ][: self.num_atoms]
        atom_types_data = [0, 1, 2, 3][: self.num_atoms]
        radii_data = [1.2, 1.5, 1.4, 1.3][: self.num_atoms]

        if torch is not None:
            coords = torch.tensor(coords_data, dtype=torch.float32)  # [N, 3]
            atom_types = torch.tensor(atom_types_data, dtype=torch.long)  # [N]
            radii = torch.tensor(radii_data, dtype=torch.float32)  # [N]
            query_points = sample_query_points(
                coords=coords,
                num_query_points=self.num_query_points,
                padding=self.bbox_padding,
            )  # [Q, 3]
        else:
            coords = coords_data
            atom_types = atom_types_data
            radii = radii_data
            query_points = sample_query_points(
                coords=coords,
                num_query_points=self.num_query_points,
                padding=self.bbox_padding,
            )

        return {
            "id": f"{self.split}-{idx}",
            "coords": coords,
            "atom_types": atom_types,
            "radii": radii,
            "query_points": query_points,
        }
