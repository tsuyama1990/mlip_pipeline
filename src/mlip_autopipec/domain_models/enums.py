from enum import StrEnum


class ComponentRole(StrEnum):
    GENERATOR = "generator"
    ORACLE = "oracle"
    TRAINER = "trainer"
    DYNAMICS = "dynamics"
    VALIDATOR = "validator"

class GeneratorType(StrEnum):
    MOCK = "mock"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

class OracleType(StrEnum):
    MOCK = "mock"
    QE = "qe"

class TrainerType(StrEnum):
    MOCK = "mock"
    PACE = "pace"

class DynamicsType(StrEnum):
    MOCK = "mock"
    LAMMPS = "lammps"

class ValidatorType(StrEnum):
    MOCK = "mock"
    STANDARD = "standard"

class WorkflowStage(StrEnum):
    EXPLORE = "explore"
    LABEL = "label"
    TRAIN = "train"

class TaskType(StrEnum):
    SINGLE_POINT = "single_point"
    MD = "md"
    RELAX = "relax"
