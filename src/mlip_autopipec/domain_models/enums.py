from enum import StrEnum


class ComponentType(StrEnum):
    """Base class for component types (string-based)."""
    MOCK = "mock"

class GeneratorType(StrEnum):
    MOCK = "mock"
    ADAPTIVE = "adaptive"
    RANDOM = "random"

class OracleType(StrEnum):
    MOCK = "mock"
    QE = "qe"
    VASP = "vasp"

class TrainerType(StrEnum):
    MOCK = "mock"
    PACEMAKER = "pacemaker"

class DynamicsType(StrEnum):
    MOCK = "mock"
    LAMMPS = "lammps"

class ValidatorType(StrEnum):
    MOCK = "mock"
    STANDARD = "standard"
