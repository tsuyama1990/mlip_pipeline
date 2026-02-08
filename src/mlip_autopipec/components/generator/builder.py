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


class BulkBuilder(StructureBuilder):
    """Builder for bulk structures."""

    def build(self, n_structures: int, config: AdaptiveGeneratorConfig) -> Iterator[Structure]:
        # Config is strictly typed, so these checks are technically redundant if we trust Pydantic,
        # but good for runtime safety if config is manually constructed or if fields are optional in schema.
        # In AdaptiveGeneratorConfig, they are required fields (str, not Optional[str]),
        # so we can rely on Pydantic validation. However, keeping logic for now.

        # Pydantic handles required fields validation on model creation.

        for _ in range(n_structures):
            # Generate bulk structure
            # We use basic parameters from config.
            atoms = bulk(config.element, crystalstructure=config.crystal_structure, cubic=True)

            # Supercell
            if config.supercell_dim > 1:
                atoms = atoms.repeat(
                    (config.supercell_dim, config.supercell_dim, config.supercell_dim)
                )  # type: ignore[no-untyped-call]

            # Add metadata
            atoms.info["type"] = "bulk"
            atoms.info["generator"] = "BulkBuilder"

            yield Structure.from_ase(atoms)


class SurfaceBuilder(StructureBuilder):
    """Builder for surface structures."""

    def build(self, n_structures: int, config: AdaptiveGeneratorConfig) -> Iterator[Structure]:
        # Validated by Pydantic schema

        indices_pool = config.surface_indices

        for _ in range(n_structures):
            # Pick random surface index from pool
            idx = indices_pool[np.random.choice(len(indices_pool))]

            # Create base bulk first
            # Surfaces need a bulk reference.
            bulk_atoms = bulk(config.element, crystalstructure=config.crystal_structure, cubic=True)

            # Create surface
            surf = surface(bulk_atoms, tuple(idx), 3, vacuum=config.vacuum)  # type: ignore[no-untyped-call]

            # Repeat surface to make it larger in x/y if needed
            # Usually surfaces are small in x/y unless supercell specified
            if config.supercell_dim > 1:
                surf = surf.repeat((config.supercell_dim, config.supercell_dim, 1))

            surf.info["type"] = "surface"
            surf.info["generator"] = "SurfaceBuilder"
            surf.info["miller_index"] = str(tuple(idx))

            yield Structure.from_ase(surf)
