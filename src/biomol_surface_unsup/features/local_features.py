from __future__ import annotations


def build_local_features(sample: dict[str, object]) -> dict[str, object]:
    values = list(sample.get("values", []))
    return {"count": len(values), "sum": float(sum(values))}
