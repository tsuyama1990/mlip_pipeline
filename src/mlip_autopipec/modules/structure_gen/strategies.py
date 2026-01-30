import ase.build
from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure


class ColdStartStrategy:
    """Strategy to generate initial structures from scratch."""

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a bulk structure based on configuration.
        """
        # Create bulk crystal
        # cubic=True ensures we get a cubic cell if possible, which is easier for MD
        atoms = ase.build.bulk(
            config.element,
            crystalstructure=config.crystal_structure,
            a=config.lattice_constant,
            cubic=True,
        )  # type: ignore[no-untyped-call]

        # Create supercell
        atoms *= config.supercell  # type: ignore[no-untyped-call]

        return Structure.from_ase(atoms)


class RattleStrategy:
    """Strategy to apply random thermal noise (rattle) to a structure."""

    def apply(self, structure: Structure, stdev: float, seed: int = 42) -> Structure:
        """
        Apply random displacement to atoms.

        Args:
            structure: Input structure.
            stdev: Standard deviation of the Gaussian noise (Angstrom).
            seed: Random seed.

        Returns:
            New rattled structure.
        """
        if stdev <= 0:
            return structure

        atoms = structure.to_ase()
        atoms.rattle(stdev=stdev, seed=seed)  # type: ignore[no-untyped-call]
        return Structure.from_ase(atoms)
