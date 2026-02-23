"""Base Oracle implementation."""

from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING, TypeVar

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import Oracle
from pyacemaker.core.utils import validate_structure_integrity
from pyacemaker.domain_models.models import (
    StructureMetadata,
    StructureStatus,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class BaseOracle(Oracle):
    """Base implementation for Oracle modules with common utilities."""

    def __init__(self, config: PYACEMAKERConfig) -> None:
        """Initialize the BaseOracle."""
        super().__init__(config)
        self.logger = logger.bind(name=self.__class__.__name__)

    def validate_structure(self, structure: StructureMetadata) -> None:
        """Validate structure metadata before processing."""
        if not isinstance(structure, StructureMetadata):
            msg = f"Expected StructureMetadata, got {type(structure).__name__}"
            raise TypeError(msg)
        validate_structure_integrity(structure)

    def _extract_atoms(self, structure: StructureMetadata) -> Atoms | None:
        """Extract ASE Atoms object from structure metadata."""
        atoms = structure.features.get("atoms")
        try:
            from ase import Atoms

            if isinstance(atoms, Atoms):
                return atoms
        except ImportError:
            pass

        self.logger.warning(f"Structure {structure.id} has no valid 'atoms' feature.")
        return None

    def _validate_and_extract_atoms(self, structure: StructureMetadata) -> Atoms | None:
        """Validate structure and extract atoms in one go.

        Returns:
            Atoms object if valid and not already calculated, None otherwise (and sets status).
        """
        self.validate_structure(structure)
        if structure.status == StructureStatus.CALCULATED:
            return None

        atoms = self._extract_atoms(structure)
        if atoms is None:
            structure.status = StructureStatus.FAILED
            return None
        return atoms

    def _process_in_batches(
        self,
        iterator: Iterable[T],
        batch_processor: Callable[[list[T]], Iterable[T] | None],
        batch_size: int = 100,
    ) -> Iterator[T]:
        """Process an iterator in batches.

        Args:
            iterator: Input iterable.
            batch_processor: Function that processes a list of items.
                             Should return an iterable of results or None.
            batch_size: Size of batches.

        Yields:
            Processed items.
        """
        iterator = iter(iterator)
        while True:
            chunk = list(islice(iterator, batch_size))
            if not chunk:
                break

            results = batch_processor(chunk)
            if results is not None:
                yield from results
            else:
                # If processor returns None, it might have modified chunk in-place or failed
                # We yield the original chunk items (which might be updated)
                yield from chunk
