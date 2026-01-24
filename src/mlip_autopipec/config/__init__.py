"""
Configuration Module.

This module defines the Pydantic schemas for the system configuration.
It serves as the contract for all other modules, ensuring type safety and validation.
"""

# EmbeddingConfig is imported in models.py now, so we can export it if models.py exports it.
# But models.py doesn't export it in its namespace properly unless I aliased it or added to __all__ there.
# I added EmbeddingConfig to models.py imports.
# Let's see if I can import it.
# Actually, better to only export what is really in models.py or what is needed.
from mlip_autopipec.config.schemas.common import EmbeddingConfig

from .models import (
    DFTConfig,
    GeneratorConfig,
    InferenceConfig,
    MinimalConfig,
    SurrogateConfig,
    SystemConfig,
    TrainingConfig,
)

__all__ = [
    "DFTConfig",
    "EmbeddingConfig",
    "GeneratorConfig",
    "InferenceConfig",
    "MinimalConfig",
    "SurrogateConfig",
    "SystemConfig",
    "TrainingConfig",
]
