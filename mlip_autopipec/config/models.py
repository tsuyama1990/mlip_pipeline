from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TargetSystem(BaseModel):
    elements: list[str]
    composition: dict[str, float]

    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    @classmethod
    def check_elements(cls, v: list[str]) -> list[str]:
        for el in v:
            if el not in chemical_symbols:
                msg = f"Invalid chemical symbol: {el}"
                raise ValueError(msg)
        return v

    @field_validator("composition")
    @classmethod
    def check_composition(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            msg = f"Composition must sum to 1.0, got {total}"
            raise ValueError(msg)
        return v


class Resources(BaseModel):
    dft_code: Literal["quantum_espresso", "vasp"]
    parallel_cores: int = Field(gt=0)
    gpu_enabled: bool = False

    model_config = ConfigDict(extra="forbid")


class MinimalConfig(BaseModel):
    project_name: str
    target_system: TargetSystem
    resources: Resources

    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    minimal: MinimalConfig
    working_dir: Path
    db_path: Path
    log_path: Path

    model_config = ConfigDict(frozen=True, extra="forbid")
