from enum import StrEnum, auto


class ComponentRole(StrEnum):
    GENERATOR = auto()
    ORACLE = auto()
    TRAINER = auto()
    DYNAMICS = auto()
    VALIDATOR = auto()
    ORCHESTRATOR = auto()

class TaskStatus(StrEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

class GeneratorType(StrEnum):
    RANDOM = auto()
    MOCK = auto()
    M3GNET = auto()
    MD = auto()
    MC = auto()
    ADAPTIVE = auto()

class OracleType(StrEnum):
    MOCK = auto()
    QE = auto()
    VASP = auto()
    CP2K = auto()

class TrainerType(StrEnum):
    MOCK = auto()
    PACE = auto()

class DynamicsType(StrEnum):
    MOCK = auto()
    LAMMPS = auto()
    EON = auto()

class ValidatorType(StrEnum):
    MOCK = auto()
    STANDARD = auto()
