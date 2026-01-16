"""Module for the Physics-Informed Generator."""

import logging

import numpy as np
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config_schemas import SystemConfig

logger = logging.getLogger(__name__)


class PhysicsInformedGenerator:
    """A factory for generating diverse initial sets of atomic structures."""

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the PhysicsInformedGenerator."""
        self.config = config

    def generate(self) -> list[Atoms]:
        """Generate a set of atomic structures based on the configuration."""
        logger.info("Starting structure generation...")
        num_elements = len(self.config.dft.input.pseudopotentials)
        if num_elements > 1:
            logger.info("Detected multiple elements, running MOCK alloy generation workflow.")
            return self._generate_for_alloy()
        logger.info("Detected single element, running MOCK crystal defect generation workflow.")
        return self._generate_for_crystal()

    def _generate_for_alloy(self) -> list[Atoms]:
        """Generate a mock structure for an alloy system."""
        base_structure = self._create_mock_alloy_structure()
        strained_structures = self._apply_strains(base_structure)
        all_base_structures = [base_structure, *strained_structures]

        final_structures = []
        for structure in all_base_structures:
            rattled_structures = self._apply_rattling(structure)
            final_structures.extend([structure, *rattled_structures])

        logger.info("Generated %d MOCK alloy structures.", len(final_structures))
        return final_structures

    def _create_mock_alloy_structure(self) -> Atoms:
        """Create a mock alloy structure."""
        logger.warning("Creating MOCK alloy structure due to dependency issues.")
        atoms: Atoms = bulk("Cu", "fcc", a=4.0).repeat((2, 2, 2))
        symbols = ["Cu"] * 4 + ["Au"] * 4
        atoms.set_chemical_symbols(symbols)
        return atoms

    def _apply_strains(self, atoms: Atoms) -> list[Atoms]:
        """Apply a series of isotropic strains to an Atoms object."""
        strained_atoms_list = []
        original_cell = atoms.get_cell()
        strain_magnitudes = self.config.generator.alloy_params.strain_magnitudes

        for strain in strain_magnitudes:
            if np.isclose(strain, 1.0):
                continue
            strained_atoms = atoms.copy()
            strained_atoms.set_cell(original_cell * strain, scale_atoms=True)
            strained_atoms_list.append(strained_atoms)
        logger.info("Applied %d strain levels.", len(strained_atoms_list))
        return strained_atoms_list

    def _apply_rattling(self, atoms: Atoms) -> list[Atoms]:
        """Apply random atomic displacements to an Atoms object."""
        rattled_atoms_list = []
        std_devs = self.config.generator.alloy_params.rattle_std_devs

        for std_dev in std_devs:
            rattled_atoms = atoms.copy()
            rattled_atoms.rattle(stdev=std_dev, seed=42)
            rattled_atoms_list.append(rattled_atoms)
        logger.info("Applied %d rattle levels.", len(rattled_atoms_list))
        return rattled_atoms_list

    def _generate_for_crystal(self) -> list[Atoms]:
        """Generate a mock structure for a crystalline system with defects."""
        final_structures = []
        element = next(iter(self.config.dft.input.pseudopotentials.keys()))
        pristine_supercell: Atoms = bulk(element, "fcc", a=4.0, cubic=True).repeat((3, 3, 3))

        defect_types = self.config.generator.crystal_params.defect_types
        if "vacancy" in defect_types:
            final_structures.append(self._create_vacancy(pristine_supercell))

        logger.info("Generated %d MOCK crystal defect structures.", len(final_structures))
        return final_structures

    def _create_vacancy(self, atoms: Atoms) -> Atoms:
        """Create a single vacancy in an Atoms object."""
        vacancy_atoms = atoms.copy()
        del vacancy_atoms[0]
        logger.info("Created a single vacancy structure.")
        return vacancy_atoms
