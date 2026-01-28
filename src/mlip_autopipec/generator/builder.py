import logging

from ase import Atoms
from ase.build import bulk, make_supercell

from mlip_autopipec.config.schemas.generator import StructureGenerationConfig
from mlip_autopipec.generator.defects import DefectStrategy
from mlip_autopipec.generator.transformations import TransformationStrategy

logger = logging.getLogger(__name__)

class StructureBuilder:
    def __init__(self, config: StructureGenerationConfig):
        self.config = config

    def build(self) -> list[Atoms]:
        structures = []
        base_atoms = bulk("Al", "fcc", a=4.05)

        if self.config.supercell:
            base_atoms = make_supercell(base_atoms, self.config.supercell_matrix)

        structures.append(base_atoms)

        if self.config.transformations.enabled:
            strategy = TransformationStrategy(self.config.transformations)
            structures.extend(strategy.apply(base_atoms))

        if self.config.defects.enabled:
            strategy = DefectStrategy(self.config.defects)
            structures.extend(strategy.apply(base_atoms))

        valid_structures = []
        for atoms in structures:
            if self._validate(atoms):
                valid_structures.append(atoms)

        return valid_structures

    def _validate(self, atoms: Atoms) -> bool:
        if not isinstance(atoms, Atoms):
            logger.error("Generated object is not an ASE Atoms object")
            return False

        if hasattr(atoms, "positions"):
            import numpy as np
            if np.isnan(atoms.positions).any() or np.isinf(atoms.positions).any():
                logger.error("Structure has NaN/Inf positions")
                return False

        return True
