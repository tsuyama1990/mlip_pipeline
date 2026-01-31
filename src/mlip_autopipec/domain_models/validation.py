from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ValidationMetric(BaseModel):
    """
    Represents a single validation metric (e.g., Bulk Modulus).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    reference: Optional[float] = None
    error: Optional[float] = None
    message: Optional[str] = None
    passed: bool


class ValidationResult(BaseModel):
    """
    Aggregated result of all validation tests.
    """

    model_config = ConfigDict(extra="forbid")

    potential_id: str
    metrics: List[ValidationMetric]
    plots: Dict[str, Path]
    overall_status: Literal["PASS", "WARN", "FAIL"]
