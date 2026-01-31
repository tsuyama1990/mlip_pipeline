from typing import Protocol

import ase.build
from mlip_autopipec.domain_models.config import (
    BulkStructureGenConfig,
    DefectStructureGenConfig,
    RandomSliceStructureGenConfig,
    StrainStructureGenConfig,
    StructureGenConfig,
)
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

        Args:
            config: A BulkStructureGenConfig containing parameters like element,
                    crystal structure, lattice constant, and supercell size.

        Returns:
            A Structure object representing the generated bulk material.
        """
        if not isinstance(config, BulkStructureGenConfig):
             raise TypeError(f"BulkStructureGenerator received incompatible config: {type(config)}")

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


class RandomSliceGenerator:
    """Strategy for generating random slice (surface/slab) structures."""

    def __init__(self) -> None:
        pass

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a random slice/slab structure.

        This strategy creates a bulk structure and cuts it along a random plane
        or specific Miller indices (if implemented), adding vacuum padding.
        """
        if not isinstance(config, RandomSliceStructureGenConfig):
            raise TypeError(f"RandomSliceGenerator received incompatible config: {type(config)}")

        # Placeholder implementation
        # Ideally: generate bulk, then slice with random Miller indices
        # For now, we'll just generate a bulk and warn or return it
        # Real implementation would go here.

        atoms = ase.build.bulk(
            name=config.element,
            crystalstructure=config.crystal_structure,
            a=config.lattice_constant,
            cubic=True
        )
        atoms = atoms * config.supercell
        # Add vacuum?
        atoms.center(vacuum=config.vacuum, axis=2) # Slab

        return Structure.from_ase(atoms)


class DefectGenerator:
    """Strategy for generating structures with defects."""

    def __init__(self) -> None:
        pass

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a structure containing a point defect.

        This strategy loads a base structure and introduces a vacancy,
        interstitial, or substitutional defect based on the configuration.
        """
        if not isinstance(config, DefectStructureGenConfig):
             raise TypeError(f"DefectGenerator received incompatible config: {type(config)}")

        # Placeholder
        # Need to load base structure from path
        # And introduce defect
        raise NotImplementedError("Defect generation strategy not yet implemented")


class StrainGenerator:
    """Strategy for generating strained structures."""

    def __init__(self) -> None:
        pass

    def generate(self, config: StructureGenConfig) -> Structure:
        """
        Generate a strained structure for elasticity calculations.

        This strategy loads a base structure and applies a deformation gradient
        defined by the strain configuration.
        """
        if not isinstance(config, StrainStructureGenConfig):
             raise TypeError(f"StrainGenerator received incompatible config: {type(config)}")

        # Placeholder
        raise NotImplementedError("Strain generation strategy not yet implemented")
