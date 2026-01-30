from typing import Protocol

import ase.build
from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder


class StructureGenerator(Protocol):
    """Protocol for structure generation strategies."""

    def generate(self, config: StructureGenConfig) -> Structure:
        """Generate a structure based on configuration."""
        ...


class BulkStructureGenerator:
    """Strategy for generating bulk crystal structures."""

    def __init__(self) -> None:
        self.builder = StructureBuilder()

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a bulk structure using ASE and apply configuration settings.
        """
        # Generate base bulk structure
        atoms = ase.build.bulk(
            name=config.element,
            crystalstructure=config.crystal_structure,
            a=config.lattice_constant,
            cubic=True,
        )

        # Apply supercell expansion
        if config.supercell != (1, 1, 1):
            atoms = atoms * config.supercell

        structure = Structure.from_ase(atoms)

        # Apply thermal noise (rattling) if requested
        if config.rattle_stdev > 0:
            structure = self.builder.apply_rattle(structure, config.rattle_stdev)

        return structure
