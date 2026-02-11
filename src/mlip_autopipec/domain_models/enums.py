from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class GeneratorType(StrEnum):
    MOCK = "MOCK"
    RANDOM = "RANDOM"
    M3GNET = "M3GNET"

class OracleType(StrEnum):
    MOCK = "MOCK"
    VASP = "VASP"
    QE = "QE"

class TrainerType(StrEnum):
    MOCK = "MOCK"
    ACE = "ACE"
    M3GNET = "M3GNET"

class DynamicsType(StrEnum):
    MOCK = "MOCK"
    LAMMPS = "LAMMPS"
