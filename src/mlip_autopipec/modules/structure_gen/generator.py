from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.modules.structure_gen.strategies import ColdStartStrategy, RattleStrategy


class StructureGenerator:
    """
    Facade for structure generation logic.
    Handles creation of initial structures for MD or training.
    """

    def __init__(self, config: StructureGenConfig):
        self.config = config
        self._cold_start = ColdStartStrategy()
        self._rattle = RattleStrategy()

    def build_initial_structure(self, seed: int = 42) -> Structure:
        """
        Builds an initial structure based on configuration.
        Applies rattling if configured.

        Args:
            seed: Random seed for rattling.

        Returns:
            Generated Structure.
        """
        structure = self._cold_start.generate(self.config)

        if self.config.rattle_stdev > 0:
            structure = self._rattle.apply(structure, self.config.rattle_stdev, seed)

        return structure
