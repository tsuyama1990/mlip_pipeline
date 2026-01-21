from pydantic import BaseModel, ConfigDict


class SurrogateConfig(BaseModel):
    """Configuration for surrogate model."""
    model_path: str | None = None
    model_config = ConfigDict(extra="forbid")

class EmbeddingConfig(BaseModel):
    # Redundant but referenced in some imports, keep sync or alias
    # Actually, Inference usually owns embedding.
    # But if SPEC says surrogate has it...
    # I'll define it here if needed, or import.
    # For now, simplistic definition.
    core_radius: float = 4.0
    model_config = ConfigDict(extra="forbid")
