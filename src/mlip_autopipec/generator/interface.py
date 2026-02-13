import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ase import Atoms

from mlip_autopipec.domain_models.datastructures import Structure

if TYPE_CHECKING:
    from mlip_autopipec.domain_models.config import GeneratorConfig

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract Base Class for Structure Generators."""

    @abstractmethod
    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        """
        Generates candidate structures.

        Args:
            context: Dictionary containing exploration parameters.
                Common keys:
                - 'cycle' (int): Current active learning cycle.
                - 'temperature' (float): Target temperature for generation.
                - 'count' (int): Number of structures to generate (optional).
                - 'mode' (str): Generation mode (e.g., 'seed' for OTF).

        Returns:
            An iterator of Structure objects.
        """

    def generate_local_candidates(
        self, structure: Structure, count: int = 5
    ) -> Iterator[Structure]:
        """
        Generates local candidates around a seed structure (e.g., for active learning).
        Default implementation returns just the seed to support subclasses that don't implement this yet.

        Args:
            structure: The seed structure (e.g. from a halt).
            count: Number of candidates to generate.

        Returns:
            An iterator of local candidate structures.
        """
        logger.warning(
            "Using default generate_local_candidates (no-op). Subclasses should override."
        )
        yield structure


class MockGenerator(BaseGenerator):
    """Mock implementation of Structure Generator."""

    def __init__(self, config: "GeneratorConfig | None" = None) -> None:
        self.config = config

    def explore(self, context: dict[str, Any]) -> Iterator[Structure]:
        logger.info("MockGenerator: Generating random structures...")

        # Validate context
        self._validate_context(context)

        # Make number configurable, default to 2
        default_count = self.config.mock_count if self.config else 2
        count = context.get("count", default_count)

        if isinstance(count, int):
            for i in range(count):
                # Create a simple dimer with cell to ensure valid LAMMPS input
                atoms = Atoms(
                    "He2",
                    positions=[[0, 0, 0], [0, 0, 1.5 + i * 0.1]],
                    cell=[10.0, 10.0, 10.0],
                    pbc=True,
                )
                yield Structure(atoms=atoms, provenance="mock_generator", label_status="unlabeled")
        else:
            # Fallback if not int
            pass

    def _validate_context(self, context: dict[str, Any]) -> None:
        """Validates exploration context to prevent injection and ensuring types."""
        allowed_keys = {"cycle", "temperature", "count", "mode"}
        for key, value in context.items():
            if key not in allowed_keys:
                msg = f"Unexpected key in exploration context: {key}"
                raise ValueError(msg)

            self._validate_key(key, value)

    def _validate_key(self, key: str, value: Any) -> None:
        """Helper to validate individual keys."""
        if key == "cycle":
            self._validate_cycle(value)
        elif key == "temperature":
            self._validate_temperature(value)
        elif key == "count":
            self._validate_count(value)
        elif key == "mode":
            self._validate_mode(value)

    def _validate_cycle(self, value: Any) -> None:
        if not isinstance(value, int):
            msg = f"Context 'cycle' must be int, got {type(value)}"
            raise TypeError(msg)
        if value < 0:
            msg = f"Context 'cycle' must be non-negative, got {value}"
            raise ValueError(msg)

    def _validate_temperature(self, value: Any) -> None:
        if not isinstance(value, (int, float)):
            msg = f"Context 'temperature' must be number, got {type(value)}"
            raise TypeError(msg)
        if value <= 0:
            msg = f"Context 'temperature' must be positive, got {value}"
            raise ValueError(msg)

    def _validate_count(self, value: Any) -> None:
        if not isinstance(value, int):
            msg = f"Context 'count' must be int, got {type(value)}"
            raise TypeError(msg)
        if value <= 0:
            msg = f"Context 'count' must be positive, got {value}"
            raise ValueError(msg)

    def _validate_mode(self, value: Any) -> None:
        if not isinstance(value, str):
            msg = f"Context 'mode' must be str, got {type(value)}"
            raise TypeError(msg)

        # Strict whitelist validation
        allowed_modes = {"seed", "random", "adaptive"}
        if value not in allowed_modes:
            msg = f"Context 'mode' must be one of {allowed_modes}, got {value}"
            raise ValueError(msg)

    def generate_local_candidates(
        self, structure: Structure, count: int = 5
    ) -> Iterator[Structure]:
        logger.info(f"MockGenerator: Generating {count} local candidates...")
        original_atoms = structure.to_ase()
        import numpy as np

        for i in range(count):
            atoms = original_atoms.copy()  # type: ignore[no-untyped-call]
            # Simple random rattle
            positions = atoms.get_positions()
            positions += np.random.normal(0, 0.05, positions.shape)
            atoms.set_positions(positions)

            yield Structure(
                atoms=atoms, provenance=f"local_candidate_{i}", label_status="unlabeled"
            )
