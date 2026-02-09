from typing import Literal, Union, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pathlib import Path
from enum import StrEnum

from mlip_autopipec.domain_models.enums import (
    GeneratorType,
    OracleType,
    TrainerType,
    DynamicsType,
    ValidatorType,
)

# Constants for validation
MAX_CONFIG_SIZE_BYTES = 1024 * 1024  # 1MB
ALLOWED_CALCULATORS = ["espresso", "vasp", "aims"]

class CalculatorType(StrEnum):
    ESPRESSO = "espresso"
    VASP = "vasp"
    AIMS = "aims"

class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class OrchestratorConfig(BaseConfig):
    work_dir: Path
    max_cycles: int = Field(ge=1, default=10)
    uncertainty_threshold: float = Field(ge=0.0, default=5.0)

    @field_validator("work_dir")
    def validate_work_dir(cls, v: Path) -> Path:
        # Basic path safety check
        if ".." in str(v):
            raise ValueError("Path traversal (..) not allowed in work_dir")
        return v

# --- Generator Configs ---
class RandomGeneratorConfig(BaseConfig):
    type: Literal[GeneratorType.RANDOM] = GeneratorType.RANDOM
    seed: int = 42

class AdaptiveGeneratorConfig(BaseConfig):
    type: Literal[GeneratorType.ADAPTIVE] = GeneratorType.ADAPTIVE
    initial_structures_count: int = Field(ge=1, default=10)
    md_ratio: float = Field(ge=0.0, le=1.0, default=0.5)

GeneratorConfig = Union[RandomGeneratorConfig, AdaptiveGeneratorConfig]

# --- Oracle Configs ---
class MockOracleConfig(BaseConfig):
    type: Literal[OracleType.MOCK] = OracleType.MOCK

class DFTOracleConfig(BaseConfig):
    type: Literal[OracleType.DFT] = OracleType.DFT
    calculator_type: CalculatorType = CalculatorType.ESPRESSO
    command: str = "mpirun -np 4 pw.x"

    @field_validator("command")
    def validate_command(cls, v: str) -> str:
        # Basic shell injection prevention
        forbidden_chars = [";", "&&", "||", "`", "$("]
        if any(char in v for char in forbidden_chars):
            raise ValueError("Command contains potentially unsafe shell characters")
        return v

OracleConfig = Union[MockOracleConfig, DFTOracleConfig]

# --- Trainer Configs ---
class MockTrainerConfig(BaseConfig):
    type: Literal[TrainerType.MOCK] = TrainerType.MOCK

class PacemakerTrainerConfig(BaseConfig):
    type: Literal[TrainerType.PACEMAKER] = TrainerType.PACEMAKER
    cutoff: float = Field(gt=0.0, default=5.0)
    basis_size: Dict[str, int] = Field(default_factory=lambda: {"max_deg": 6, "max_body": 3})
    physics_baseline: Optional[str] = None

TrainerConfig = Union[MockTrainerConfig, PacemakerTrainerConfig]

# --- Dynamics Configs ---
class MockDynamicsConfig(BaseConfig):
    type: Literal[DynamicsType.MOCK] = DynamicsType.MOCK

class LAMMPSDynamicsConfig(BaseConfig):
    type: Literal[DynamicsType.LAMMPS] = DynamicsType.LAMMPS
    timestep: float = Field(gt=0.0, default=0.001)
    max_workers: int = Field(ge=1, default=1)
    input_filename: str = "in.lammps"
    log_filename: str = "lammps.log"
    driver_filename: str = "driver.py"

class EONDynamicsConfig(BaseConfig):
    type: Literal[DynamicsType.EON] = DynamicsType.EON
    max_workers: int = Field(ge=1, default=1)
    input_filename: str = "config.ini"
    log_filename: str = "client.log"
    driver_filename: str = "driver.py"

DynamicsConfig = Union[MockDynamicsConfig, LAMMPSDynamicsConfig, EONDynamicsConfig]

# --- Validator Configs ---
class MockValidatorConfig(BaseConfig):
    type: Literal[ValidatorType.MOCK] = ValidatorType.MOCK

class StandardValidatorConfig(BaseConfig):
    type: Literal[ValidatorType.STANDARD] = ValidatorType.STANDARD
    phonon_displacement: float = Field(gt=0.0, default=0.01)
    eos_strain_range: float = Field(gt=0.0, default=0.1)

ValidatorConfig = Union[MockValidatorConfig, StandardValidatorConfig]

# --- Root Config ---
class Config(BaseConfig):
    orchestrator: OrchestratorConfig
    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    validator: ValidatorConfig
