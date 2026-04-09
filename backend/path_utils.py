from __future__ import annotations

from pathlib import Path


def project_root_from(current_file: str | Path) -> Path:
    """Resolve the repository root by walking parents until a backend directory is found."""
    current_path = Path(current_file).resolve()
    start_dir = current_path if current_path.is_dir() else current_path.parent

    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "backend").is_dir():
            return candidate

    raise RuntimeError(f"Could not resolve project root from: {current_file}")


def models_dir_from(current_file: str | Path) -> Path:
    return project_root_from(current_file) / "models"
