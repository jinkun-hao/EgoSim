from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "video",
    "ego_prior_video",
    "hand_keypoint_video",
    "first_frame",
    "prompt",
    "task_name",
    "part_idx",
    "process_result_dir",
    "hdf5_path",
    "gt_process_result_dir",
}


def validate_metadata_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

    missing = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {', '.join(missing)}"
        )
