from __future__ import annotations

import unittest

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset


class DatasetTestCase(unittest.TestCase):
    def test_dataset_returns_toy_sample(self) -> None:
        dataset = MoleculeDataset(num_query_points=6)
        sample = dataset[0]

        self.assertEqual(sample["id"], "train-0")

        coords = sample["coords"]  # [N, 3]
        atom_types = sample["atom_types"]  # [N]
        radii = sample["radii"]  # [N]
        query_points = sample["query_points"]  # [Q, 3]

        if hasattr(coords, "shape"):
            self.assertEqual(tuple(coords.shape), (4, 3))
            self.assertEqual(tuple(atom_types.shape), (4,))
            self.assertEqual(tuple(radii.shape), (4,))
            self.assertEqual(tuple(query_points.shape), (6, 3))
        else:
            self.assertEqual(len(coords), 4)
            self.assertEqual(len(coords[0]), 3)
            self.assertEqual(len(atom_types), 4)
            self.assertEqual(len(radii), 4)
            self.assertEqual(len(query_points), 6)
            self.assertEqual(len(query_points[0]), 3)

    def test_collate_preserves_single_sample_shapes(self) -> None:
        sample = MoleculeDataset(num_query_points=5)[0]
        batch = collate_fn([sample])

        self.assertEqual(tuple(batch["coords"].shape), (4, 3))
        self.assertEqual(tuple(batch["atom_types"].shape), (4,))
        self.assertEqual(tuple(batch["radii"].shape), (4,))
        self.assertEqual(tuple(batch["query_points"].shape), (5, 3))


if __name__ == "__main__":
    unittest.main()
