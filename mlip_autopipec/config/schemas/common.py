from pydantic import BaseModel, ConfigDict, Field, field_validator


class TargetSystem(BaseModel):
    elements: list[str] = Field(..., description="List of chemical symbols")
    composition: dict[str, float] = Field(..., description="Atomic fractions")
    crystal_structure: str | None = Field(None, description="Base structure (e.g., 'fcc')")

    model_config = ConfigDict(extra="forbid")

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, elements: list[str]) -> list[str]:
        from ase.data import chemical_symbols

        for symbol in elements:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return elements

    @field_validator("composition")
    @classmethod
    def validate_composition(cls, composition: dict[str, float]) -> dict[str, float]:
        if not composition:
            raise ValueError("Composition cannot be empty.")

        # Check values range
        for sym, frac in composition.items():
            if not (0.0 <= frac <= 1.0):
                raise ValueError(f"Composition fraction for {sym} must be between 0.0 and 1.0.")

        total = sum(composition.values())
        if not (0.99999 <= total <= 1.00001):
            raise ValueError(f"Composition fractions must sum to 1.0 (got {total}).")

        from ase.data import chemical_symbols

        for symbol in composition:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol in composition.")

        return composition
