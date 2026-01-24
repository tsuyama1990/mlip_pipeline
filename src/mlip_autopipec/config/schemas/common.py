from pydantic import BaseModel, ConfigDict, Field, field_validator


class TargetSystem(BaseModel):
    name: str = Field("System", description="Name of the target system")
    elements: list[str] = Field(..., description="List of chemical symbols")
    composition: dict[str, float] = Field(..., description="Atomic fractions")
    crystal_structure: str | None = Field(None, description="Base structure (e.g., 'fcc')")

    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, elements: list[str]) -> list[str]:
        cls._validate_symbols(elements)
        return elements

    @field_validator("composition")
    @classmethod
    def validate_composition(cls, composition: dict[str, float]) -> dict[str, float]:
        cls._validate_composition_not_empty(composition)
        cls._validate_composition_range(composition)
        cls._validate_composition_sum(composition)
        cls._validate_symbols(list(composition.keys()))
        return composition

    @classmethod
    def _validate_symbols(cls, symbols: list[str]) -> None:
        from ase.data import chemical_symbols
        for symbol in symbols:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")

    @classmethod
    def _validate_composition_not_empty(cls, composition: dict[str, float]) -> None:
        if not composition:
            raise ValueError("Composition cannot be empty.")

    @classmethod
    def _validate_composition_range(cls, composition: dict[str, float]) -> None:
        for sym, frac in composition.items():
            if not (0.0 <= frac <= 1.0):
                raise ValueError(f"Composition fraction for {sym} must be between 0.0 and 1.0.")

    @classmethod
    def _validate_composition_sum(cls, composition: dict[str, float]) -> None:
        total = sum(composition.values())
        if not (0.99999 <= total <= 1.00001):
            raise ValueError(f"Composition fractions must sum to 1.0 (got {total}).")


class EmbeddingConfig(BaseModel):
    """Configuration for cluster embedding."""
    core_radius: float = Field(4.0, gt=0.0)
    buffer_width: float = Field(2.0, gt=0.0)
    model_config = ConfigDict(extra="forbid")

    @property
    def box_size(self) -> float:
        return 2 * (self.core_radius + self.buffer_width) + 2.0
