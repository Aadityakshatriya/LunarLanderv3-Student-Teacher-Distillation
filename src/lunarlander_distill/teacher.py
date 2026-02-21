from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class TeacherSpec:
    repo_id: str
    filename: Optional[str] = None


def resolve_teacher_filename(repo_id: str) -> str:
    """Try to infer the SB3 zip filename from the repo if not provided."""

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    zip_files = [f for f in files if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError(
            f"No .zip files found in HF repo '{repo_id}'. Files: {files[:50]}"
        )
    # prefer a file that mentions ppo and lunarlander if present
    preferred = [
        f
        for f in zip_files
        if ("ppo" in f.lower()) and ("lunar" in f.lower() or "lander" in f.lower())
    ]
    return (preferred[0] if preferred else zip_files[0])


def download_teacher_zip(
    *,
    repo_id: str,
    filename: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    if filename is None:
        filename = resolve_teacher_filename(repo_id)
    token = token or os.environ.get("HF_TOKEN")
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return Path(path)
