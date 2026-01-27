"""
Configuration Module.

This module defines the Pydantic schemas for the system configuration.
It serves as the contract for all other modules, ensuring type safety and validation.
"""

from mlip_autopipec.config.schemas.common import EmbeddingConfig
# Import things that are used in models.py or schemas
from mlip_autopipec.config.schemas.core import RuntimeConfig, TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.config.schemas.exploration import ExplorerConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.config.schemas.workflow import WorkflowConfig

from .models import (
    MinimalConfig,
    SystemConfig,
)

__all__ = [
    "DFTConfig",
    "EmbeddingConfig",
    "ExplorerConfig",
    "GeneratorConfig",
    "InferenceConfig",
    "MinimalConfig",
    "RuntimeConfig",
    "SurrogateConfig",
    "SystemConfig",
    "TargetSystem",
    "TrainingConfig",
    "ValidationConfig",
    "WorkflowConfig",
]
