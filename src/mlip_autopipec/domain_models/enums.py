from enum import StrEnum


class ComponentRole(StrEnum):
    GENERATOR = "generator"
    ORACLE = "oracle"
    TRAINER = "trainer"
    DYNAMICS = "dynamics"
    VALIDATOR = "validator"


class GeneratorType(StrEnum):
    RANDOM = "RANDOM"
    ADAPTIVE = "ADAPTIVE"


class OracleType(StrEnum):
    QE = "QE"
    VASP = "VASP"
    MOCK = "MOCK"


class TrainerType(StrEnum):
    PACEMAKER = "PACEMAKER"
    MOCK = "MOCK"


class DynamicsType(StrEnum):
    LAMMPS = "LAMMPS"
    EON = "EON"
    MOCK = "MOCK"


class ValidatorType(StrEnum):
    STANDARD = "STANDARD"
    MOCK = "MOCK"
