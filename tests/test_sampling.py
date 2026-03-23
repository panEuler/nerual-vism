from __future__ import annotations

import unittest

from biomol_surface_unsup.datasets.sampling import sample_points


class SamplingTestCase(unittest.TestCase):
    def test_sample_points_returns_requested_count(self) -> None:
        points = sample_points(3)
        self.assertEqual(len(points), 3)


if __name__ == "__main__":
    unittest.main()
