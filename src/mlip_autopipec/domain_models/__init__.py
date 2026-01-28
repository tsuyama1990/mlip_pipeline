"""
Domain Models Module.

This module defines the core data structures used to exchange information
between pipeline stages (e.g. DFT results, Training data, Extracted structures).
These models are distinct from configuration schemas.
"""

from .dft_models import DFTResult
from .inference_models import ExtractedStructure
from .training_data import TrainingBatch, TrainingData
from .validation import ValidationMetric, ValidationResult

__all__ = [
    "DFTResult",
    "ExtractedStructure",
    "TrainingBatch",
    "TrainingData",
    "ValidationMetric",
    "ValidationResult",
]
