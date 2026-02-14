"""Utility functions for PYACEMAKER."""

from collections.abc import Iterator
from uuid import uuid4

from pyacemaker.domain_models.models import (
    MaterialDNA,
    StructureMetadata,
    StructureStatus,
)


def validate_structure_integrity(structure: StructureMetadata) -> None:
    """Validate the integrity of a structure metadata object.

    Args:
        structure: The structure metadata to validate.

    Raises:
        ValueError: If validation fails.

    """
    # Check for features dictionary keys
    if not all(isinstance(k, str) for k in structure.features):
        msg = "Structure features keys must be strings."
        raise ValueError(msg)


def generate_dummy_structures(
    count: int = 10, tags: list[str] | None = None
) -> Iterator[StructureMetadata]:
    """Generate dummy structures for testing (Lazily).

    Args:
        count: Number of structures to generate.
        tags: Optional tags to add.

    Yields:
        Dummy StructureMetadata objects.

    """
    try:
        from ase import Atoms
    except ImportError:
        Atoms_cls = None
    else:
        Atoms_cls = Atoms

    tags_list = tags or ["dummy"]

    for i in range(count):
        features = {}
        if Atoms_cls:
            features["atoms"] = Atoms_cls("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        yield StructureMetadata(
            id=uuid4(),
            tags=[f"{t}_{i}" for t in tags_list],
            status=StructureStatus.NEW,
            features=features,
            material_dna=MaterialDNA(composition={"H": 1.0}),  # Dummy DNA
        )
