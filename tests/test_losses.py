from __future__ import annotations

import unittest

from biomol_surface_unsup.losses.loss_builder import build_loss


class LossesTestCase(unittest.TestCase):
    def test_build_loss_and_call(self) -> None:
        loss_fn = build_loss("weak_prior")
        value = loss_fn({"sdf": 1.0}, {"values": [0.0]})
        self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
