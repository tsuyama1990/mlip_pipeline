from enum import Enum


class ComponentType(str, Enum):
    """Base class for component types (string-based)."""
    MOCK = "mock"

class GeneratorType(str, Enum):
    MOCK = "mock"
    ADAPTIVE = "adaptive"
    RANDOM = "random"

class OracleType(str, Enum):
    MOCK = "mock"
    QE = "qe"
    VASP = "vasp"

class TrainerType(str, Enum):
    MOCK = "mock"
    PACEMAKER = "pacemaker"

class DynamicsType(str, Enum):
    MOCK = "mock"
    LAMMPS = "lammps"

class ValidatorType(str, Enum):
    MOCK = "mock"
    STANDARD = "standard"
