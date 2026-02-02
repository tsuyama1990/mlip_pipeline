from pathlib import Path
from typing import Any, Protocol

from mlip_autopipec.domain_models.potential import Potential


class Explorer(Protocol):
    """Protocol for the Exploration module (MD/MC/Structure Generation)."""

    def explore(self, potential: Potential | None, work_dir: Path) -> dict[str, Any]:
        """
        Explores the configuration space using the given potential.

        Returns:
            A dictionary containing results of exploration (e.g., halt status, candidate files).
        """
        ...


class Oracle(Protocol):
    """Protocol for the Oracle module (DFT Calculations)."""

    def compute(self, structures: list[Path], work_dir: Path) -> list[Path]:
        """
        Performs DFT calculations on the given structures.

        Returns:
            A list of paths to the computed data files.
        """
        ...


class Trainer(Protocol):
    """Protocol for the Trainer module (Potential Fitting)."""

    def train(
        self,
        dataset: Path,
        previous_potential: Path | None,
        output_dir: Path
    ) -> Path:
        """
        Trains a new potential using the given dataset.

        Returns:
            The path to the trained potential file.
        """
        ...


class Validator(Protocol):
    """Protocol for the Validator module (Quality Assurance)."""

    def validate(self, potential: Path) -> dict[str, Any]:
        """
        Validates the given potential against a test suite.

        Returns:
            A validation report dictionary.
        """
        ...
