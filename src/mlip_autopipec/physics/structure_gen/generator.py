from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.physics.structure_gen.strategies import (
    BulkStructureGenerator,
    DefectGenerator,
    RandomSliceGenerator,
    StrainGenerator,
    StructureGenerator,
)


class StructureGenFactory:
    """Factory to create the appropriate structure generator based on config."""

    @staticmethod
    def get_generator(config: StructureGenConfig) -> StructureGenerator:
        if config.strategy == "bulk":
            return BulkStructureGenerator()
        if config.strategy == "random_slice":
            return RandomSliceGenerator()
        if config.strategy == "defect":
            return DefectGenerator()
        if config.strategy == "strain":
            return StrainGenerator()

        msg = f"Unknown structure generation strategy: {config.strategy}"
        raise ValueError(msg)
