"""
ACDC dataset wrapper.

ACDC distribution layout (after extraction of training.zip):
    data/training/
        patient001/
            Info.cfg                       # metadata (Group, NbFrame, ED, ES, ...)
            patient001_4d.nii.gz           # full 4D cine sequence (H, W, D, T)
            patient001_frame01.nii.gz      # ED frame (3D)
            patient001_frame01_gt.nii.gz   # ED ground-truth segmentation
            patient001_frame12.nii.gz      # ES frame (3D)
            patient001_frame12_gt.nii.gz   # ES ground-truth segmentation
        patient002/ ...

Disease subgroup is encoded in Info.cfg → field "Group":
    NOR  — Normal
    MINF — Previous Myocardial Infarction
    DCM  — Dilated Cardiomyopathy
    HCM  — Hypertrophic Cardiomyopathy
    RV   — Abnormal Right Ventricle  (file says "RV", we map to "ARV")

Label classes in *_gt.nii.gz volumes:
    0 — background
    1 — right ventricle cavity
    2 — myocardium
    3 — left ventricle cavity
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

GROUP_TO_INDEX: dict[str, int] = {
    "NOR": 0,
    "MINF": 1,
    "DCM": 2,
    "HCM": 3,
    "RV": 4,    # written "RV" in Info.cfg, semantic = "abnormal RV"
}


@dataclass
class ACDCPatient:
    """One patient's file paths + metadata."""

    patient_id: str
    group: str
    ed_frame: int
    es_frame: int
    image_ed: Path
    image_es: Path
    label_ed: Path
    label_es: Path
    image_4d: Path

    @property
    def class_index(self) -> int:
        return GROUP_TO_INDEX[self.group]


def _parse_info_cfg(info_path: Path) -> dict[str, str]:
    """Parse the simple key:value Info.cfg file."""
    out: dict[str, str] = {}
    for line in info_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def discover_patients(training_dir: str | Path) -> list[ACDCPatient]:
    """Walk the training directory and build an ACDCPatient list."""
    training_dir = Path(training_dir)
    patients: list[ACDCPatient] = []
    for pdir in sorted(training_dir.glob("patient*")):
        info = _parse_info_cfg(pdir / "Info.cfg")
        ed = int(info["ED"])
        es = int(info["ES"])
        patients.append(
            ACDCPatient(
                patient_id=pdir.name,
                group=info["Group"],
                ed_frame=ed,
                es_frame=es,
                image_ed=pdir / f"{pdir.name}_frame{ed:02d}.nii.gz",
                image_es=pdir / f"{pdir.name}_frame{es:02d}.nii.gz",
                label_ed=pdir / f"{pdir.name}_frame{ed:02d}_gt.nii.gz",
                label_es=pdir / f"{pdir.name}_frame{es:02d}_gt.nii.gz",
                image_4d=pdir / f"{pdir.name}_4d.nii.gz",
            )
        )
    return patients


def build_splits(
    patients: Sequence[ACDCPatient],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Build patient-stratified K-fold splits.

    Stratification key = disease group, so each fold contains a
    representative mix of NOR / MINF / DCM / HCM / RV.
    """
    y = np.array([p.class_index for p in patients])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train.tolist(), val.tolist()) for train, val in skf.split(np.zeros(len(y)), y)]


# ----------------------------------------------------------------------
# MONAI-compatible Dataset
# ----------------------------------------------------------------------

class ACDCDataset:
    """
    Thin wrapper around monai.data.Dataset.

    Each item is a dict consumed by MONAI's dictionary transforms:
        {
            "image":     str path to a 3D volume,
            "label":     str path to a 3D segmentation,
            "patient":   patient_id,
            "phase":     "ED" or "ES",
            "group":     "NOR" / ... / "RV",
            "class":     int class index for diagnosis branch,
        }

    Build with `ACDCDataset.from_patients(patients, transform=...)`.
    """

    def __init__(self, data: list[dict], transform=None) -> None:
        # Lazy import so the module is importable without MONAI installed
        from monai.data import Dataset as MonaiDataset
        self._ds = MonaiDataset(data=data, transform=transform)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]

    @classmethod
    def from_patients(
        cls,
        patients: Sequence[ACDCPatient],
        transform=None,
        include_es: bool = True,
    ) -> "ACDCDataset":
        items: list[dict] = []
        for p in patients:
            items.append(
                dict(
                    image=str(p.image_ed),
                    label=str(p.label_ed),
                    patient=p.patient_id,
                    phase="ED",
                    group=p.group,
                    **{"class": p.class_index},
                )
            )
            if include_es:
                items.append(
                    dict(
                        image=str(p.image_es),
                        label=str(p.label_es),
                        patient=p.patient_id,
                        phase="ES",
                        group=p.group,
                        **{"class": p.class_index},
                    )
                )
        return cls(items, transform=transform)
