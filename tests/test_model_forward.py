from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.models.surface_model import SurfaceModel


def test_model_forward_single_sample_keeps_compatibility():
    dataset = MoleculeDataset(num_query_points=32)
    sample = dataset[0]

    model = SurfaceModel(num_atom_types=16)
    out = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
    )
    assert out["sdf"].shape == (32,)
    assert out["features"].shape[:2] == (32, 4)
    assert out["mask"].shape[:2] == (32, 4)


def test_model_forward_batched_uses_atom_and_query_masks():
    dataset = MoleculeDataset(num_samples=2, num_atoms=4, num_query_points=8)
    batch = collate_fn([dataset[0], dataset[1]])

    model = SurfaceModel(num_atom_types=16)
    out = model(
        batch["coords"],
        batch["atom_types"],
        batch["radii"],
        batch["query_points"],
        atom_mask=batch["atom_mask"],
        query_mask=batch["query_mask"],
    )
    assert out["sdf"].shape == (2, 8)
    assert out["features"].shape[:3] == (2, 8, 4)
    assert out["mask"].shape == (2, 8, 4)
    assert torch.all(out["sdf"][~batch["query_mask"]] == 0.0)
    assert not torch.any(out["mask"][1, :, 3])
