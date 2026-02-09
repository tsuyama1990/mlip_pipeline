from typing import Literal, Union, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pathlib import Path
from enum import StrEnum
import shlex

from mlip_autopipec.domain_models.enums import (
    GeneratorType,
    OracleType,
    TrainerType,
    DynamicsType,
    ValidatorType,
)
from mlip_autopipec.constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_EON_TIME_STEP,
    DEFAULT_LAMMPS_INPUT,
    DEFAULT_LAMMPS_LOG,
    DEFAULT_LAMMPS_DRIVER,
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
        # Prevent traversal and ensure path is safe
        try:
            # We want to allow creating new directories, so we check the parent if it exists
            # Or just check for ".." traversal relative to current
            # A strict check: resolve() throws if not exists on some platforms/versions,
            # but usually it's fine for non-existent paths on modern Pythons (>=3.6).
            # However, resolve() resolves symlinks.
            # Let's check for ".." in parts
            if ".." in v.parts:
                 raise ValueError("Path traversal (..) not allowed in work_dir")

            # If path exists, check if it resolves to a sensitive system dir?
            # For now, ".." check is the primary defense against relative traversal.
            return v
        except Exception as e:
            raise ValueError(f"Invalid work_dir: {e}")

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
    batch_size: int = Field(ge=1, default=DEFAULT_BUFFER_SIZE)

    @field_validator("command")
    def validate_command(cls, v: str) -> str:
        # Sanitization: Ensure command is parseable by shlex and doesn't contain chaining
        try:
            tokens = shlex.split(v)
            if not tokens:
                raise ValueError("Command cannot be empty")

            # Check for dangerous operators that shlex handles but are risky in shell=True
            # If we assume shell=False (recommended), we just need arguments.
            # But the user might expect shell features.
            # Strictly forbidding chaining operators:
            forbidden_operators = [";", "&&", "||", "|", "&", ">", "<", "`", "$("]
            if any(op in v for op in forbidden_operators):
                 raise ValueError(f"Command contains forbidden shell operators: {forbidden_operators}")

            return v
        except ValueError as e:
            raise ValueError(f"Invalid command format: {e}")

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
    input_filename: str = DEFAULT_LAMMPS_INPUT
    log_filename: str = DEFAULT_LAMMPS_LOG
    driver_filename: str = DEFAULT_LAMMPS_DRIVER

class EONDynamicsConfig(BaseConfig):
    type: Literal[DynamicsType.EON] = DynamicsType.EON
    max_workers: int = Field(ge=1, default=1)
    input_filename: str = "config.ini"
    log_filename: str = "client.log"
    driver_filename: str = "driver.py"
    time_step: float = Field(gt=0.0, default=DEFAULT_EON_TIME_STEP)

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
