from abc import ABC, abstractmethod

import numpy as np
from ase.build import bulk, surface

from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.structure import Structure


class StructureBuilder(ABC):
    """Abstract base class for structure builders."""

    @abstractmethod
    def build(self, n_structures: int, config: GeneratorConfig) -> list[Structure]:
        """
        Build a list of structures.

        Args:
            n_structures: Number of structures to generate.
            config: Configuration object.

        Returns:
            List of generated Structure objects.
        """
        ...


class BulkBuilder(StructureBuilder):
    """Builder for bulk structures."""

    def build(self, n_structures: int, config: GeneratorConfig) -> list[Structure]:
        if not config.element:
            msg = "Element must be specified for BulkBuilder"
            raise ValueError(msg)

        structures = []
        for _ in range(n_structures):
            # Generate bulk structure
            # We use basic parameters from config.
            atoms = bulk(config.element, crystalstructure=config.crystal_structure, cubic=True)

            # Supercell
            if config.supercell_dim > 1:
                atoms = atoms.repeat((config.supercell_dim, config.supercell_dim, config.supercell_dim)) # type: ignore[no-untyped-call]

            # Add metadata
            atoms.info["type"] = "bulk"
            atoms.info["generator"] = "BulkBuilder"

            structures.append(Structure.from_ase(atoms))

        return structures


class SurfaceBuilder(StructureBuilder):
    """Builder for surface structures."""

    def build(self, n_structures: int, config: GeneratorConfig) -> list[Structure]:
        if not config.element:
            msg = "Element must be specified for SurfaceBuilder"
            raise ValueError(msg)

        structures = []
        indices_pool = config.surface_indices

        for _ in range(n_structures):
            # Pick random surface index from pool
            idx =  indices_pool[np.random.choice(len(indices_pool))] # type: ignore

            # Create base bulk first
            # Surfaces need a bulk reference.
            bulk_atoms = bulk(config.element, crystalstructure=config.crystal_structure, cubic=True)

            # Create surface
            surf = surface(bulk_atoms, tuple(idx), 3, vacuum=config.vacuum) # type: ignore[no-untyped-call]

            # Repeat surface to make it larger in x/y if needed
            # Usually surfaces are small in x/y unless supercell specified
            if config.supercell_dim > 1:
                 surf = surf.repeat((config.supercell_dim, config.supercell_dim, 1)) # type: ignore[no-untyped-call]

            surf.info["type"] = "surface"
            surf.info["generator"] = "SurfaceBuilder"
            surf.info["miller_index"] = str(tuple(idx))

            structures.append(Structure.from_ase(surf))

        return structures
