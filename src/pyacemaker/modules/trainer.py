"""Trainer (Pacemaker) module implementation."""

from typing import Any

from pyacemaker.trainer.mace_trainer import MaceTrainer
from pyacemaker.trainer.pacemaker import PacemakerTrainer

try:
    from ase import Atoms
except ImportError:
    Atoms = Any  # type: ignore[assignment,misc]

__all__ = ["MaceTrainer", "PacemakerTrainer"]
