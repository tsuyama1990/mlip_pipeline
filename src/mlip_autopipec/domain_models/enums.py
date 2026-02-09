from enum import StrEnum


class ComponentRole(StrEnum):
    """Enum for component roles."""

    GENERATOR = "generator"
    ORACLE = "oracle"
    TRAINER = "trainer"
    DYNAMICS = "dynamics"
    VALIDATOR = "validator"

    def __repr__(self) -> str:
        return f"<ComponentRole.{self.name}>"

    def __str__(self) -> str:
        return self.value


class ComponentType(StrEnum):
    """Base class for component types (string-based)."""

    MOCK = "mock"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}>"

    def __str__(self) -> str:
        return self.value


class GeneratorType(StrEnum):
    MOCK = "mock"
    ADAPTIVE = "adaptive"

    def __repr__(self) -> str:
        return f"<GeneratorType.{self.name}>"

    def __str__(self) -> str:
        return self.value


class OracleType(StrEnum):
    MOCK = "mock"
    QE = "qe"
    VASP = "vasp"

    def __repr__(self) -> str:
        return f"<OracleType.{self.name}>"

    def __str__(self) -> str:
        return self.value


class TrainerType(StrEnum):
    MOCK = "mock"
    PACEMAKER = "pacemaker"

    def __repr__(self) -> str:
        return f"<TrainerType.{self.name}>"

    def __str__(self) -> str:
        return self.value


class DynamicsType(StrEnum):
    MOCK = "mock"
    LAMMPS = "lammps"
    EON = "eon"

    def __repr__(self) -> str:
        return f"<DynamicsType.{self.name}>"

    def __str__(self) -> str:
        return self.value


class ValidatorType(StrEnum):
    MOCK = "mock"
    STANDARD = "standard"

    def __repr__(self) -> str:
        return f"<ValidatorType.{self.name}>"

    def __str__(self) -> str:
        return self.value
