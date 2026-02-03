from typing import Protocol

from ase import Atoms


class StructureGenerator(Protocol):
    """Protocol for structure generation strategies."""

    def generate(self, atoms: Atoms, count: int) -> list[Atoms]:
        """Generates new structures from a seed structure."""
        ...
