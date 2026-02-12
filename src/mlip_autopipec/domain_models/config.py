from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.enums import GeneratorType, OracleType, TrainerType


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrchestratorConfig(BaseConfig):
    work_dir: Path
    max_iterations: int = Field(ge=1, default=10)


class RandomGeneratorConfig(BaseConfig):
    type: Literal["RANDOM"] = GeneratorType.RANDOM
    num_structures: int = Field(default=10, ge=1)


class M3GNetGeneratorConfig(BaseConfig):
    type: Literal["M3GNET"] = GeneratorType.M3GNET
    model_path: str = Field(default="DEFAULT")


GeneratorConfig = RandomGeneratorConfig | M3GNetGeneratorConfig


class DFTOracleConfig(BaseConfig):
    type: Literal["QUANTUM_ESPRESSO"] = OracleType.QUANTUM_ESPRESSO
    command: str = Field(min_length=1)
    mixing_beta: float = Field(default=0.7, ge=0.0, le=1.0)


OracleConfig = DFTOracleConfig


class PacemakerTrainerConfig(BaseConfig):
    type: Literal["PACEMAKER"] = TrainerType.PACEMAKER
    r_cut: float = Field(default=5.0, ge=1.0)
    max_deg: int = Field(default=3, ge=1)


TrainerConfig = PacemakerTrainerConfig


class FullConfig(BaseConfig):
    orchestrator: OrchestratorConfig
    generator: GeneratorConfig = Field(discriminator="type")
    oracle: OracleConfig
    trainer: TrainerConfig
