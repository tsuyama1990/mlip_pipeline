from pydantic import BaseModel, ConfigDict, Field
from mlip_autopipec.domain_models.validation import ValidationResult

class ProductionManifest(BaseModel):
    """
    Metadata for a production-ready potential.
    """
    model_config = ConfigDict(extra="forbid")

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic Versioning (X.Y.Z)")
    author: str
    training_set_size: int = Field(default=0, ge=0)
    validation_metrics: ValidationResult
    license: str = "MIT"
    description: str = "Auto-generated MLIP potential"
