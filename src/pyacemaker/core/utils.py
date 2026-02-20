"""Utility functions for PYACEMAKER."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from pyacemaker.domain_models.models import (
    MaterialDNA,
    StructureMetadata,
    StructureStatus,
)

if TYPE_CHECKING:
    from ase import Atoms


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


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest of SHA256 checksum.

    """
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify if file matches expected checksum.

    Args:
        file_path: Path to the file.
        expected_checksum: Expected SHA256 hex digest.

    Returns:
        True if checksum matches, False otherwise.

    """
    if not file_path.exists():
        return False
    return calculate_checksum(file_path) == expected_checksum


def atoms_to_metadata(atoms: "Atoms", **kwargs: Any) -> StructureMetadata:
    """Convert ASE Atoms object to StructureMetadata.

    Args:
        atoms: The ASE Atoms object.
        **kwargs: Additional fields for StructureMetadata.

    Returns:
        StructureMetadata object wrapping the atoms.

    """
    # Create basic metadata
    features = kwargs.pop("features", {})
    features["atoms"] = atoms

    # Extract DNA
    composition = _extract_composition(atoms)

    # Extract Properties
    energy, forces, stress, status = _extract_properties(atoms, kwargs)

    if energy is not None and forces is not None:
        status = StructureStatus.CALCULATED

    return StructureMetadata(
        features=features,
        material_dna=MaterialDNA(composition=composition),
        energy=energy,
        forces=forces,
        stress=stress,
        status=status,
        **kwargs,
    )


def _extract_composition(atoms: "Atoms") -> dict[str, float]:
    """Helper to extract composition from atoms."""
    composition = {}
    try:
        symbols = atoms.get_chemical_symbols()  # type: ignore[no-untyped-call]
        total = len(symbols)
        if total > 0:
            for s in set(symbols):
                composition[s] = symbols.count(s) / total
    except Exception as e:
        logger.debug(f"Failed to extract composition: {e}")
    return composition


def _extract_properties(
    atoms: "Atoms", kwargs: dict[str, Any]
) -> tuple[float | None, list[list[float]] | None, list[float] | None, StructureStatus]:
    """Helper to extract energy, forces, stress from atoms."""
    energy = kwargs.pop("energy", None)
    forces = kwargs.pop("forces", None)
    stress = kwargs.pop("stress", None)
    status = kwargs.pop("status", StructureStatus.NEW)

    try:
        # Check explicit calc results
        calc = getattr(atoms, "calc", None)
        if calc:
            if energy is None and "energy" in calc.results:
                energy = float(calc.results["energy"])
            if forces is None and "forces" in calc.results:
                forces = calc.results["forces"].tolist()
            if stress is None and "stress" in calc.results:
                stress = calc.results["stress"].tolist()

        # Also check info/arrays as fallback
        if energy is None and "energy" in atoms.info:
            energy = float(atoms.info["energy"])
        if forces is None and "forces" in atoms.arrays:
            forces = atoms.arrays["forces"].tolist()

    except Exception as e:
        logger.debug(f"Failed to extract properties: {e}")

    return energy, forces, stress, status


def metadata_to_atoms(metadata: StructureMetadata) -> "Atoms":
    """Extract ASE Atoms object from StructureMetadata.

    Also attaches calculation results (energy, forces) to the Atoms object
    using SinglePointCalculator so they are preserved during serialization.

    Args:
        metadata: The structure metadata.

    Returns:
        The ASE Atoms object found in features (updated with results).

    Raises:
        ValueError: If 'atoms' is missing from features.

    """
    if "atoms" not in metadata.features:
        msg = "StructureMetadata does not contain 'atoms' in features."
        raise ValueError(msg)

    atoms = metadata.features["atoms"]

    # Attach results if present
    if metadata.energy is not None and metadata.forces is not None:
        try:
            from ase.calculators.singlepoint import SinglePointCalculator

            # Prepare results dict
            results = {
                "energy": metadata.energy,
                "forces": metadata.forces,
            }
            if metadata.stress is not None:
                results["stress"] = metadata.stress

            atoms.calc = SinglePointCalculator(atoms, **results)  # type: ignore[no-untyped-call]

            # Also store in info for redundancy
            atoms.info["energy"] = metadata.energy

        except ImportError:
            logger.warning("ASE SinglePointCalculator not available.")
        except Exception as e:
            logger.warning(f"Failed to attach results to atoms: {e}")

    return atoms  # type: ignore[no-any-return]
