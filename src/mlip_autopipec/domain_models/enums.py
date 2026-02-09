from enum import StrEnum, auto


class ComponentRole(StrEnum):
    GENERATOR = auto()
    ORACLE = auto()
    TRAINER = auto()
    DYNAMICS = auto()
    VALIDATOR = auto()


class GeneratorType(StrEnum):
    RANDOM = auto()
    ADAPTIVE = auto()


class OracleType(StrEnum):
    DFT = auto()
    MOCK = auto()


class TrainerType(StrEnum):
    PACEMAKER = auto()
    MOCK = auto()


class DynamicsType(StrEnum):
    LAMMPS = auto()
    EON = auto()
    MOCK = auto()


class ValidatorType(StrEnum):
    STANDARD = auto()
    MOCK = auto()
