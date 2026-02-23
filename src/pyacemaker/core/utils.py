"""Utility functions for PYACEMAKER."""

import contextlib
import hashlib
import heapq
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

import numpy as np
from loguru import logger

from pyacemaker.domain_models.models import (
    MaterialDNA,
    StructureMetadata,
    StructureStatus,
)

if TYPE_CHECKING:
    from ase import Atoms

T = TypeVar("T")


def select_top_k_structures(
    iterator: Iterable[T], k: int, key_func: Callable[[T], float]
) -> list[T]:
    """Select the top K elements from an iterable based on a key function.

    Uses a min-heap of size K to maintain the top elements in O(K) memory.
    This avoids materializing the full iterator which heapq.nlargest might do depending on implementation.

    Args:
        iterator: Iterable of items.
        k: Number of items to select.
        key_func: Function to extract comparison key (larger is better).

    Returns:
        List of selected items (sorted by key descending).

    """
    if k <= 0:
        return []

    # Min-heap stores tuples of (key, item).
    # We want top K largest keys.
    # If heap size < k: push.
    # If heap size == k: pushpop if new key > min key in heap.
    heap: list[tuple[float, int, T]] = []

    # Tie-breaker counter to ensure stability/comparability if T is not comparable
    # and to handle duplicate keys deterministically.
    for counter, item in enumerate(iterator):
        key = key_func(item)
        entry = (key, counter, item)

        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif key > heap[0][0]:
            heapq.heapreplace(heap, entry)

    # Sort by key descending
    return [item for _, _, item in sorted(heap, key=lambda x: x[0], reverse=True)]


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

    # Validate atoms if present
    if "atoms" in structure.features:
        validate_structure_integrity_atoms(structure.features["atoms"])

    # Validate consistency of forces if present
    if structure.forces is not None and "atoms" in structure.features:
        atoms = structure.features["atoms"]
        if len(structure.forces) != len(atoms):
            msg = f"Forces array length ({len(structure.forces)}) does not match atom count ({len(atoms)})"
            raise ValueError(msg)


def validate_structure_integrity_atoms(atoms: "Atoms") -> None:
    """Validate an ASE Atoms object.

    Args:
        atoms: The ASE Atoms object to validate.

    Raises:
        ValueError: If validation fails.

    """
    from ase import Atoms

    if not isinstance(atoms, Atoms):
        msg = f"Expected ASE Atoms object, got {type(atoms).__name__}"
        raise TypeError(msg)

    if len(atoms) == 0:
        msg = "Structure contains no atoms"
        raise ValueError(msg)

    if not hasattr(atoms, "numbers") or not hasattr(atoms, "positions"):
        msg = "Structure missing essential attributes (numbers, positions)"
        raise ValueError(msg)

    # Check for NaN/Inf in positions
    if np.isnan(atoms.positions).any() or np.isinf(atoms.positions).any():
        msg = "Structure positions contain NaN or Inf values"
        raise ValueError(msg)

    # Check cell if periodic
    if atoms.pbc.any():
        if np.isnan(atoms.cell).any() or np.isinf(atoms.cell).any():
            msg = "Structure cell contains NaN or Inf values"
            raise ValueError(msg)

        # Check for zero volume (singular cell) if fully periodic
        if atoms.pbc.all():
            try:
                # use get_volume() if available, else determinant
                vol = atoms.get_volume()  # type: ignore[no-untyped-call]
            except Exception:
                # Fallback if get_volume fails (e.g. rank < 3)
                vol = 0.0

            if abs(vol) < 1e-6:
                msg = "Structure cell volume is near zero or invalid"
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
    # Validate atoms first
    validate_structure_integrity_atoms(atoms)

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
            energy, forces, stress = _extract_from_calc(calc, energy, forces, stress)

        # Also check info/arrays as fallback
        energy, forces = _extract_from_info_arrays(atoms, energy, forces)

        _extract_additional_metadata(atoms, kwargs)

    except Exception as e:
        logger.debug(f"Failed to extract properties: {e}")

    return energy, forces, stress, status


def _extract_from_calc(
    calc: Any, energy: float | None, forces: list | None, stress: list | None
) -> tuple[float | None, list | None, list | None]:
    if energy is None and "energy" in calc.results:
        energy = float(calc.results["energy"])
    if forces is None and "forces" in calc.results:
        forces = calc.results["forces"].tolist()
    if stress is None and "stress" in calc.results:
        stress = calc.results["stress"].tolist()
    return energy, forces, stress


def _extract_from_info_arrays(
    atoms: "Atoms", energy: float | None, forces: list | None
) -> tuple[float | None, list | None]:
    if energy is None and "energy" in atoms.info:
        energy = float(atoms.info["energy"])
    if forces is None and "forces" in atoms.arrays:
        forces = atoms.arrays["forces"].tolist()
    return energy, forces


def _extract_additional_metadata(atoms: "Atoms", kwargs: dict[str, Any]) -> None:
    for key in ["label_source", "generation_method"]:
        if key in atoms.info and key not in kwargs:
            kwargs[key] = atoms.info[key]

    if "tags" in atoms.info and "tags" not in kwargs:
        tags_val = atoms.info["tags"]
        if isinstance(tags_val, str):
            kwargs["tags"] = tags_val.split(",")
        elif isinstance(tags_val, list):
            kwargs["tags"] = tags_val


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
    # Validate metadata first
    validate_structure_integrity(metadata)

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

    # Store persistent metadata in info
    if metadata.label_source:
        atoms.info["label_source"] = metadata.label_source
    if metadata.generation_method:
        atoms.info["generation_method"] = metadata.generation_method
    if metadata.tags:
        atoms.info["tags"] = metadata.tags

    return atoms  # type: ignore[no-any-return]


def _validate_result_atoms(result_atoms: "Atoms") -> None:
    """Validate that the result atoms object has required methods."""
    if not hasattr(result_atoms, "get_potential_energy"):
        msg = "Result atoms object missing energy calculation method"
        raise TypeError(msg)


def update_structure_metadata(structure: StructureMetadata, result_atoms: "Atoms | None") -> None:
    """Update structure metadata with results (Energy, Forces, Stress).

    Args:
        structure: The structure metadata object to update.
        result_atoms: The ASE Atoms object containing calculation results.

    """
    if result_atoms is None:
        structure.status = StructureStatus.FAILED
        return

    try:
        # Validate input atoms
        _validate_result_atoms(result_atoms)

        # We use type: ignore because ASE types are often missing
        energy_val = result_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        energy = float(energy_val)

        forces_arr = result_atoms.get_forces()  # type: ignore[no-untyped-call]
        forces: list[list[float]] = forces_arr.tolist()

        stress: list[float] | None = None
        with contextlib.suppress(Exception):
            stress_arr = result_atoms.get_stress()  # type: ignore[no-untyped-call]
            stress = stress_arr.tolist()

        # Update explicit fields
        structure.energy = energy
        structure.forces = forces
        if stress:
            structure.stress = stress

        structure.features["atoms"] = result_atoms
        # Set status last
        structure.status = StructureStatus.CALCULATED

    except Exception:
        logger.exception(f"Failed to extract properties for {structure.id}")
        structure.status = StructureStatus.FAILED
