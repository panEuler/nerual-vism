from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset


def test_dataset_returns_toy_sample_with_sampling_metadata() -> None:
    torch.manual_seed(0)
    dataset = MoleculeDataset(num_query_points=8)
    sample = dataset[0]

    assert sample["id"] == "train-0"
    assert tuple(sample["coords"].shape) == (4, 3)
    assert tuple(sample["atom_types"].shape) == (4,)
    assert tuple(sample["radii"].shape) == (4,)
    assert tuple(sample["query_points"].shape) == (8, 3)
    assert tuple(sample["query_group"].shape) == (8,)
    assert tuple(sample["containment_points"].shape) == (2, 3)
    assert sample["sampling_counts"] == {"global": 4, "containment": 2, "surface_band": 2}


def test_collate_fn_pads_atoms_and_queries_without_dropping_samples() -> None:
    torch.manual_seed(0)
    dataset = MoleculeDataset(num_samples=2, num_atoms=4, num_query_points=8)
    batch = collate_fn([dataset[0], dataset[1]])

    assert batch["id"] == ["train-0", "train-1"]
    assert tuple(batch["coords"].shape) == (2, 4, 3)
    assert tuple(batch["atom_types"].shape) == (2, 4)
    assert tuple(batch["radii"].shape) == (2, 4)
    assert tuple(batch["atom_mask"].shape) == (2, 4)
    assert tuple(batch["query_points"].shape) == (2, 8, 3)
    assert tuple(batch["query_group"].shape) == (2, 8)
    assert tuple(batch["query_mask"].shape) == (2, 8)
    assert batch["atom_mask"][0].sum().item() == 4
    assert batch["atom_mask"][1].sum().item() == 3
    assert batch["query_mask"][0].sum().item() == 8
    assert batch["query_mask"][1].sum().item() == 7
    assert batch["sampling_counts"] == {"global": 7, "containment": 3, "surface_band": 5}
