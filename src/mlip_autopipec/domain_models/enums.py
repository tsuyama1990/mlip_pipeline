from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ComponentRole(StrEnum):
    GENERATOR = "GENERATOR"
    ORACLE = "ORACLE"
    TRAINER = "TRAINER"
    DYNAMICS = "DYNAMICS"
    VALIDATOR = "VALIDATOR"


class GeneratorType(StrEnum):
    RANDOM = "RANDOM"
    M3GNET = "M3GNET"


class OracleType(StrEnum):
    CP2K = "CP2K"
    VASP = "VASP"
    QUANTUM_ESPRESSO = "QUANTUM_ESPRESSO"
    MACE_MP = "MACE_MP"


class TrainerType(StrEnum):
    PACEMAKER = "PACEMAKER"
    MACE = "MACE"
