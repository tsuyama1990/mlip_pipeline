"""Common definitions for domain models."""

from datetime import UTC, datetime
from enum import StrEnum


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class StructureStatus(StrEnum):
    """Status of a structure in the active learning cycle."""

    NEW = "NEW"
    CALCULATED = "CALCULATED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class CycleStatus(StrEnum):
    """Status of the active learning cycle."""

    # Standard AL Statuses
    EXPLORATION = "EXPLORATION"
    LABELING = "LABELING"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    CONVERGED = "CONVERGED"
    FAILED = "FAILED"

    # MACE Distillation Workflow Statuses
    DIRECT_SAMPLING = "DIRECT_SAMPLING"
    MACE_TRAINING = "MACE_TRAINING"
    SURROGATE_GENERATION = "SURROGATE_GENERATION"
    SURROGATE_LABELING = "SURROGATE_LABELING"
    DELTA_LEARNING = "DELTA_LEARNING"


class PotentialType(StrEnum):
    """Type of interatomic potential."""

    PACE = "PACE"
    MACE = "MACE"
    M3GNET = "M3GNET"
    LJ = "LJ"
    EAM = "EAM"


class TaskType(StrEnum):
    """Type of computational task."""

    DFT = "DFT"
    TRAINING = "TRAINING"
    MD = "MD"
    KMC = "KMC"


class TaskStatus(StrEnum):
    """Status of a task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
