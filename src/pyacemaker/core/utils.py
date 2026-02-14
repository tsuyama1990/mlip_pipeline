"""Utility functions for PYACEMAKER."""

from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

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


def get_structure_id(atoms: Any) -> UUID:
    """Extract or generate UUID for an Atoms object.

    Args:
        atoms: An ASE Atoms object.

    Returns:
        UUID: The unique identifier.

    """
    if not hasattr(atoms, "info"):
        return uuid4()

    uid_str = atoms.info.get("uuid")
    if uid_str:
        try:
            return UUID(str(uid_str))
        except ValueError:
            pass

    # Generate new if missing or invalid
    new_id = uuid4()
    atoms.info["uuid"] = str(new_id)
    return new_id


def metadata_to_atoms(metadata: StructureMetadata) -> Any:
    """Convert StructureMetadata to ASE Atoms with full fidelity."""
    validate_structure_integrity(metadata)
    atoms = metadata.features.get("atoms")
    if atoms is None:
        msg = f"Structure {metadata.id} does not contain 'atoms' feature."
        raise ValueError(msg)

    # Create a copy to avoid modifying original
    atoms = atoms.copy()

    # Inject UUID
    atoms.info["uuid"] = str(metadata.id)

    # Inject Energy/Forces/Stress if available (overwrite calc results)
    if metadata.energy is not None:
        atoms.info["energy"] = metadata.energy
    if metadata.forces is not None:
        # arrays expects numpy array or list
        atoms.arrays["forces"] = metadata.forces
    if metadata.stress is not None:
        atoms.info["stress"] = metadata.stress

    # Inject full metadata for round-trip fidelity
    # We exclude 'features' because it contains the atoms object itself (recursion)
    import contextlib

    with contextlib.suppress(Exception):
        atoms.info["_metadata_json"] = metadata.model_dump_json(exclude={"features"})

    return atoms


def atoms_to_metadata(atoms: Any) -> StructureMetadata:
    """Reconstruct StructureMetadata from ASE Atoms."""
    # Try to reconstruct full metadata from JSON if available
    if "_metadata_json" in atoms.info:
        import contextlib

        with contextlib.suppress(Exception):
            meta = StructureMetadata.model_validate_json(atoms.info["_metadata_json"])
            # Re-attach the atoms object
            meta.features["atoms"] = atoms
            return meta

    # Fallback to minimal reconstruction
    uid = get_structure_id(atoms)

    meta = StructureMetadata(
        id=uid,
        features={"atoms": atoms},
        energy=atoms.info.get("energy"),
    )
    if "forces" in atoms.arrays:
        meta.forces = atoms.arrays["forces"].tolist()
    if "stress" in atoms.info:
        stress_val = atoms.info["stress"]
        meta.stress = stress_val.tolist() if hasattr(stress_val, "tolist") else stress_val
    return meta


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
