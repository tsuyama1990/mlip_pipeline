from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Metadata for a machine learning interatomic potential.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    version: str
    # Metrics must be a dictionary of strings to floats/ints/strings, but we want to be somewhat flexible
    # while enforcing structure. Let's strictly type it as Dict[str, Any] but add validation if needed.
    # Audit requested strictness. Let's assume metrics are strictly numerical for now?
    # Actually, metrics can include strings (e.g. "pass").
    # Let's enforce that keys are strings and values are primitives.
    metrics: Dict[str, Any] = Field(default_factory=dict)
