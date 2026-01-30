from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.modules.structure_gen.strategies import (
    ColdStartStrategy,
    RattleStrategy,
)


class StructureGenerator:
    def __init__(self, config: StructureGenConfig, seed: int = 42):
        self.config = config
        self.seed = seed

    def build(self) -> Structure:
        # 1. Generate base
        cold_start = ColdStartStrategy()
        structure = cold_start.generate(
            element=self.config.element,
            crystal_structure=self.config.crystal_structure,
            lattice_constant=self.config.lattice_constant,
            supercell=self.config.supercell,
        )

        # 2. Rattle
        if self.config.rattle_stdev > 0:
            rattler = RattleStrategy(stdev=self.config.rattle_stdev, seed=self.seed)
            structure = rattler.apply(structure)

        return structure
