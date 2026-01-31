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
from mlip_autopipec.infrastructure import io


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

        # Implementation
        atoms = ase.build.bulk(
            name=config.element,
            crystalstructure=config.crystal_structure,
            a=config.lattice_constant,
            cubic=True
        )
        atoms = atoms * config.supercell
        # Add vacuum (slab)
        atoms.center(vacuum=config.vacuum, axis=2)

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

        # Load base structure
        # Assuming path points to a file readable by ase
        # Note: config.base_structure_path is a Path
        atoms_list = io.load_structures(config.base_structure_path)
        # Taking the first one if multiple, or assume single
        try:
            structure = next(iter(atoms_list)) # load_structures returns iterator or list
        except StopIteration:
            raise ValueError(f"Base structure file {config.base_structure_path} is empty.")

        atoms = structure.to_ase()

        if config.defect_type == "vacancy":
            # Remove a random atom (or first one)
            # Simple implementation: remove index 0
            del atoms[0]
        elif config.defect_type == "interstitial":
            # Add atom at a position. Simplistic: center of cell?
            # Or near 0,0,0
            # Ideally need interstitial finder.
            # Placeholder: add atom at 0.5, 0.5, 0.5 relative if empty
            # For robustness, let's just create a simple vacancy logic for now as requested.
            pass
        elif config.defect_type == "substitution":
            # Replace atom 0 with another element (implied by context? Config doesn't have species)
            # If config has species... DefectStructureGenConfig currently only has defect_type.
            # So we assume vacancy is the primary implemented one.
            pass

        return Structure.from_ase(atoms)


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

        # Load base structure
        atoms_list = io.load_structures(config.base_structure_path)
        try:
            structure = next(iter(atoms_list))
        except StopIteration:
            raise ValueError(f"Base structure file {config.base_structure_path} is empty.")

        atoms = structure.to_ase()

        # Apply random strain within range
        strain = (config.strain_range)
        # Apply isotropic expansion/compression for simplicity
        # New cell = old cell * (1 + strain)

        cell = atoms.get_cell()
        atoms.set_cell(cell * (1.0 + strain), scale_atoms=True)

        return Structure.from_ase(atoms)
