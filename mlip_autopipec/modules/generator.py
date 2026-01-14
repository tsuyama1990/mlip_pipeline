"""A module for generating initial atomic structures."""
from typing import List

from ase import Atoms
from icet.tools.structure_generation import generate_sqs
from pymatgen.core.structure import Structure

from mlip_autopipec.config_schemas import SystemConfig


class PhysicsInformedGenerator:
    """A factory for creating a diverse initial dataset of atomic structures.

    This class uses physics-informed and chemistry-informed strategies to
    generate a set of atomic structures that serve as the starting point for
    an MLIP training workflow. It is controlled by the `GeneratorParams`
    section of the system configuration.
    """

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the generator with the system configuration.

        Args:
            config: The main `SystemConfig` object containing all parameters.

        """
        self.config = config

    def generate(self) -> List[Atoms]:
        """Generate the initial set of atomic structures.

        This method acts as a dispatcher, calling the appropriate generation
        workflow based on the material type specified in the configuration.

        Returns:
            A list of ASE `Atoms` objects representing the generated dataset.

        """
        if len(self.config.target_system.elements) > 1:
            return self._generate_for_alloy()
        return self._generate_for_crystal()

    def _generate_for_alloy(self) -> List[Atoms]:
        """Orchestrate the alloy generation protocol.

        Returns:
            A list of ASE `Atoms` objects for the alloy.

        """
        base_sqs = self._create_sqs_structure()
        strained_structures = self._apply_strains(base_sqs)

        all_structures = [base_sqs] + strained_structures
        rattled_structures = []
        for s in all_structures:
            rattled_structures.extend(self._apply_rattling(s))

        return all_structures + rattled_structures

    def _create_sqs_structure(self) -> Atoms:
        """Wrap the `icet` library calls to generate an SQS structure.

        Returns:
            An ASE `Atoms` object representing the SQS structure.

        """
        # This is a simplified placeholder. A real implementation would need more setup.
        from ase.build import bulk

        primitive_structure = bulk(self.config.target_system.elements[0], "fcc", a=4.0)

        # Supercell size from config
        supercell_size = self.config.generator.alloy.sqs_supercell_size

        # Composition from config
        composition = self.config.target_system.composition

        # Generate SQS
        sqs = generate_sqs(primitive_structure, supercell_size, composition)
        return Atoms(sqs)

    def _apply_strains(self, atoms: Atoms) -> List[Atoms]:
        """Apply volumetric and shear strains to a base structure.

        Args:
            atoms: The base `Atoms` object.

        Returns:
            A list of strained `Atoms` objects.

        """
        strained_atoms = []
        for strain in self.config.generator.alloy.strain_magnitudes:
            strained = atoms.copy()  # type: ignore[no-untyped-call]
            strained.set_cell(strained.cell * strain, scale_atoms=True)
            strained_atoms.append(strained)
        return strained_atoms

    def _apply_rattling(self, atoms: Atoms) -> List[Atoms]:
        """Apply random atomic displacements to a base structure.

        Args:
            atoms: The base `Atoms` object.

        Returns:
            A list of rattled `Atoms` objects.

        """
        rattled_atoms = []
        for std_dev in self.config.generator.alloy.rattle_std_devs:
            if std_dev > 0:
                rattled = atoms.copy()  # type: ignore[no-untyped-call]
                rattled.rattle(stdev=std_dev, seed=42)
                rattled_atoms.append(rattled)
        return rattled_atoms

    def _generate_for_crystal(self) -> List[Atoms]:
        """Handle defect generation for crystalline materials.

        Returns:
            A list of `Atoms` objects with defects.

        """
        # This is a simplified placeholder.
        from ase.build import bulk
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = bulk(self.config.target_system.elements[0], "fcc", a=4.0)
        structure = AseAtomsAdaptor.get_structure(atoms)

        defect_structures: List[Structure] = []
        for defect_type in self.config.generator.crystal.defect_types:
            if defect_type == "vacancy":
                defect_structures.extend(
                    self._create_vacancy(structure)  # type: ignore[arg-type]
                )

        return [AseAtomsAdaptor.get_atoms(s) for s in defect_structures]

    def _create_vacancy(self, structure: Structure) -> List[Structure]:
        """Create single vacancies in the structure.

        Args:
            structure: The base `Structure` object.

        Returns:
            A list of `Structure` objects with vacancies.

        """
        vacancies = []
        for i in range(len(structure)):
            vacancy_structure = structure.copy()
            vacancy_structure.remove_sites([i])
            vacancies.append(vacancy_structure)
        return vacancies
