from __future__ import annotations


class EikonalLoss:
    def __call__(self, prediction: dict[str, object], target: dict[str, object]) -> float:
        return abs(float(prediction.get("sdf", 0.0)))
