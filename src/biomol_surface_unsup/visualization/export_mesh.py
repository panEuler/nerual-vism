from __future__ import annotations

from pathlib import Path


def export_mesh(mesh: dict[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(str(mesh), encoding="utf-8")
    return output_path
