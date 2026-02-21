from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class TeacherSpec:
    path: Path


def resolve_teacher_path(path: Union[str, Path], *, must_exist: bool = True) -> Path:
    """Resolve a local teacher checkpoint path and optionally validate it exists."""
    resolved = Path(path).expanduser()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Teacher not found at {resolved}. "
            "Train a teacher with train_teacher.py or pass --teacher-path."
        )
    return resolved
