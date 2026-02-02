from pathlib import Path
from typing import Any, Protocol


class Explorer(Protocol):
    def explore(self, potential_path: Path | None, work_dir: Path) -> Any:
        """
        Runs the exploration phase (e.g., MD).
        Returns result indicating if new structures were found.
        """
        ...

class Oracle(Protocol):
    def compute(self, input_data: Any, work_dir: Path) -> Any:
        """
        Runs the oracle phase (e.g., DFT).
        Returns information about the generated dataset.
        """
        ...

class Trainer(Protocol):
    def train(self, dataset: Path, previous_potential: Path | None = None) -> Path:
        """
        Runs the training phase.
        Returns the path to the trained potential.
        """
        ...

class Validator(Protocol):
    def validate(self, potential_path: Path) -> Any:
        """
        Validates the potential.
        Returns validation metrics/status.
        """
        ...
