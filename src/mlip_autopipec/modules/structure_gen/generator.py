from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.modules.structure_gen.strategies import ColdStartStrategy, RattleStrategy


class StructureGenerator:
    """Generates atomic structures for simulation."""

    def __init__(self) -> None:
        self.strategy = ColdStartStrategy()
        self.rattle = RattleStrategy()

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a structure based on configuration.

        Uses ColdStartStrategy to build bulk crystal and RattleStrategy
        to break symmetry.
        """
        structure = self.strategy.generate(config)

        # Apply small rattle to break symmetry
        # Ideally this should be configurable
        structure = self.rattle.apply(structure, stdev=0.01)

        return structure
