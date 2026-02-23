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

from pyacemaker.core.config import CONSTANTS
from pyacemaker.domain_models.models import (
    MaterialDNA,
    StructureMetadata,
    StructureStatus,
)

if TYPE_CHECKING:
    from ase import Atoms

    from pyacemaker.oracle.dataset import DatasetManager

T = TypeVar("T")


def select_top_k_structures(
    iterator: Iterable[T], k: int, key_func: Callable[[T], float]
) -> list[T]:
    """Select the top K elements from an iterable based on a key function.

    Uses heapq.nlargest which is efficient and optimized for this use case.

    Args:
        iterator: Iterable of items.
        k: Number of items to select.
        key_func: Function to extract comparison key (larger is better).

    Returns:
        List of selected items (sorted by key descending).

    """
    if k <= 0:
        return []

    return heapq.nlargest(k, iterator, key=key_func)


def validate_structure_integrity(structure: StructureMetadata) -> None:
    """Validate the integrity of a structure metadata object.

    Args:
        structure: The structure metadata to validate.

    Raises:
        ValueError: If validation fails.

    """
    # Check for features dictionary keys using iterator to be memory safe
    for k in structure.features:
        if not isinstance(k, str):
            msg = "Structure features keys must be strings."
            raise TypeError(msg)

    # Validate atoms if present
    if "atoms" in structure.features:
        validate_structure_integrity_atoms(structure.features["atoms"])

    # Validate consistency of forces if present
    if structure.forces is not None and "atoms" in structure.features:
        atoms = structure.features["atoms"]
        if len(structure.forces) != len(atoms):
            msg = f"Forces array length ({len(structure.forces)}) does not match atom count ({len(atoms)})"
            raise ValueError(msg)


def validate_structure_integrity_atoms(atoms: Any) -> None:
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

    Creates Atoms objects only when yielded to respect memory constraints.

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
    # Use block size from CONSTANTS if available, else default to 4096
    try:
        block_size = CONSTANTS.default_buffer_size
    except Exception:
        block_size = 4096

    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(block_size), b""):
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
    # Validate atoms input type
    from ase import Atoms
    if not isinstance(atoms, Atoms):
        msg = f"Expected ASE Atoms object, got {type(atoms)}"
        raise TypeError(msg)

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
    # Inject UUID for tracking
    atoms.info["uuid"] = str(metadata.id)

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


def _validate_result_atoms(result_atoms: Any) -> None:
    """Validate that the result atoms object has required methods."""
    if not hasattr(result_atoms, "get_potential_energy"):
        msg = "Result atoms object missing energy calculation method"
        raise TypeError(msg)


def stream_metadata_to_atoms(metadata_iter: Iterable[StructureMetadata]) -> Iterator[Any]:
    """Stream StructureMetadata objects to ASE Atoms.

    Args:
        metadata_iter: Iterable of StructureMetadata.

    Yields:
        ASE Atoms objects.

    """
    for metadata in metadata_iter:
        yield metadata_to_atoms(metadata)


def update_structure_metadata(structure: StructureMetadata, result_atoms: Any) -> None:
    """Update structure metadata with results (Energy, Forces, Stress).

    Args:
        structure: The structure metadata object to update.
        result_atoms: The ASE Atoms object containing calculation results.

    """
    if result_atoms is None:
        structure.status = StructureStatus.FAILED
        return

    # Basic type check for result_atoms if ASE is installed
    try:
        from ase import Atoms
        if not isinstance(result_atoms, Atoms):
             # Don't fail hard if it's a mock, but warn?
             # _validate_result_atoms will check methods.
             pass
    except ImportError:
        pass

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


def save_metadata_stream(
    dataset_manager: "DatasetManager",
    stream: Iterator[StructureMetadata],
    path: Path,
    mode: str = "ab",
    calculate_checksum: bool = False,
) -> None:
    """Convert metadata stream to atoms and save to dataset using Manager.

    Optimized to skip expensive checksum calculation if requested (e.g. for streaming append).
    Removes stale checksum file to prevent validation failures if appending.

    Args:
        dataset_manager: Instance of DatasetManager.
        stream: Iterator of StructureMetadata.
        path: Path to the dataset file.
        mode: File open mode ('w' or 'wb' for overwrite, 'a' or 'ab' for append).
        calculate_checksum: Whether to calculate checksum of the file after writing.

    """
    atoms_stream = stream_metadata_to_atoms(stream)

    # If appending, we must invalidate the existing checksum first
    if "a" in mode:
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if checksum_path.exists():
            with contextlib.suppress(OSError):
                checksum_path.unlink()

    dataset_manager.save_iter(
        atoms_stream, path, mode=mode, calculate_checksum=calculate_checksum
    )
