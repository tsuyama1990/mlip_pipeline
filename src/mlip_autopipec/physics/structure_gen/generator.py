from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.physics.structure_gen.strategies import (
    BulkStructureGenerator,
    StructureGenerator,
)


class StructureGenFactory:
    """Factory to create the appropriate structure generator based on config."""

    @staticmethod
    def get_generator(config: StructureGenConfig) -> StructureGenerator:
        if config.strategy == "bulk":
            return BulkStructureGenerator()
        if config.strategy == "surface":
            raise NotImplementedError(
                "Surface generation strategy is not yet implemented."
            )

        # Fallback for unexpected strategy if typing doesn't catch it
        msg = f"Unknown structure generation strategy: {config.strategy}"
        raise ValueError(msg)
