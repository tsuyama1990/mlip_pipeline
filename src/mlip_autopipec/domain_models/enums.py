from enum import StrEnum


class DFTCode(StrEnum):
    QUANTUM_ESPRESSO = "quantum_espresso"
    VASP = "vasp"
    CASTEP = "castep"


class DFTTask(StrEnum):
    SCF = "scf"
    RELAX = "relax"
    VC_RELAX = "vc-relax"


class TaskType(StrEnum):
    EXPLORATION = "exploration"
    TRAINING = "training"
    DYNAMICS = "dynamics"
    VALIDATION = "validation"


class ComponentRole(StrEnum):
    GENERATOR = "generator"
    ORACLE = "oracle"
    TRAINER = "trainer"
    DYNAMICS = "dynamics"
    VALIDATOR = "validator"


class GeneratorType(StrEnum):
    MOCK = "mock"
    RANDOM = "random"
    M3GNET = "m3gnet"
    ADAPTIVE = "adaptive"


class OracleType(StrEnum):
    MOCK = "mock"
    DFT = "dft"


class TrainerType(StrEnum):
    MOCK = "mock"
    PACEMAKER = "pacemaker"


class ActiveSetMethod(StrEnum):
    MAXVOL = "maxvol"
    RANDOM = "random"
    NONE = "none"


class DynamicsType(StrEnum):
    MOCK = "mock"
    LAMMPS = "lammps"
    EON = "eon"


class ValidatorType(StrEnum):
    MOCK = "mock"
    PHYSICS = "physics"


class ExecutionMode(StrEnum):
    MOCK = "mock"
    PRODUCTION = "production"
    TEST = "test"
