from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from ase.build import bulk, surface

from mlip_autopipec.domain_models.config import AdaptiveGeneratorConfig
from mlip_autopipec.domain_models.structure import Structure


class StructureBuilder(ABC):
    """Abstract base class for structure builders."""

    @abstractmethod
    def build(self, n_structures: int, config: AdaptiveGeneratorConfig) -> Iterator[Structure]:
        """
        Build a stream of structures.

        Args:
            n_structures: Number of structures to generate.
            config: Configuration object.

        Yields:
            Generated Structure objects.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class BulkBuilder(StructureBuilder):
    """Builder for bulk structures."""

    def build(self, n_structures: int, config: AdaptiveGeneratorConfig) -> Iterator[Structure]:
        for _ in range(n_structures):
            # Generate bulk structure
            # We use basic parameters from config.
            # ase.build.bulk returns Atoms
            atoms = bulk(config.element, crystalstructure=config.crystal_structure, cubic=True)  # type: ignore[no-untyped-call]

            # Supercell
            if config.supercell_dim > 1:
                # repeat returns Atoms
                atoms = atoms.repeat(  # type: ignore[no-untyped-call]
                    (config.supercell_dim, config.supercell_dim, config.supercell_dim)
                )

            # Add metadata
            atoms.info["type"] = "bulk"
            atoms.info["generator"] = "BulkBuilder"

            yield Structure.from_ase(atoms)


class SurfaceBuilder(StructureBuilder):
    """Builder for surface structures."""

    def build(self, n_structures: int, config: AdaptiveGeneratorConfig) -> Iterator[Structure]:
        indices_pool = config.surface_indices

        for _ in range(n_structures):
            # Pick random surface index from pool
            # np.random.choice returns a single item if size is not given? No, index into list.
            idx_int = int(np.random.choice(len(indices_pool)))
            idx = indices_pool[idx_int]

            # Create base bulk first
            # Surfaces need a bulk reference.
            bulk_atoms = bulk(config.element, crystalstructure=config.crystal_structure, cubic=True)  # type: ignore[no-untyped-call]

            # Create surface
            # surface returns Atoms
            surf = surface(bulk_atoms, tuple(idx), 3, vacuum=config.vacuum)  # type: ignore[no-untyped-call]

            # Repeat surface to make it larger in x/y if needed
            if config.supercell_dim > 1:
                surf = surf.repeat((config.supercell_dim, config.supercell_dim, 1))  # type: ignore[no-untyped-call]

            surf.info["type"] = "surface"
            surf.info["generator"] = "SurfaceBuilder"
            surf.info["miller_index"] = str(tuple(idx))

            yield Structure.from_ase(surf)
