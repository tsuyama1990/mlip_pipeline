from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExplorationMethod(str, Enum):
    MD = "molecular_dynamics"
    STATIC = "static_displacement"
    AKMC = "adaptive_kmc"


class StaticParameters(BaseModel):
    strain_range: float = 0.05
    defect_type: str = "vacancy"
    model_config = ConfigDict(extra="forbid")


class MDParameters(BaseModel):
    embedding_cutoff: float = 5.0
    local_displacement_range: float = 0.05
    local_sampling_count: int = 5
    # Parameters passed to LammpsRunner.run could be dynamic, but for structure gen control:
    temperature: float = 300.0
    pressure: float = 0.0
    steps: int = 1000
    model_config = ConfigDict(extra="ignore") # Allow extra MD params for LAMMPS


class AKMCParameters(BaseModel):
    temperature: float = 300.0
    model_config = ConfigDict(extra="ignore")


class BaseTask(BaseModel):
    modifiers: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class StaticTask(BaseTask):
    method: Literal[ExplorationMethod.STATIC] = ExplorationMethod.STATIC
    parameters: StaticParameters = Field(default_factory=StaticParameters)


class MDTask(BaseTask):
    method: Literal[ExplorationMethod.MD] = ExplorationMethod.MD
    parameters: MDParameters = Field(default_factory=MDParameters)


class AKMCTask(BaseTask):
    method: Literal[ExplorationMethod.AKMC] = ExplorationMethod.AKMC
    parameters: AKMCParameters = Field(default_factory=AKMCParameters)


ExplorationTask = StaticTask | MDTask | AKMCTask
