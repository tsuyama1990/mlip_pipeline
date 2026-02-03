from pathlib import Path
from typing import Protocol

from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.domain_models.validation import ValidationResult


class Explorer(Protocol):
    def explore(self, potential_path: Path | None, work_dir: Path) -> list[CandidateStructure]:
        """
        Explores the configuration space to find candidate structures.

        Args:
            potential_path: Path to the current potential (if any).
            work_dir: Directory for exploration artifacts.

        Returns:
            A list of candidate structures found during exploration.
        """
        ...


class Selector(Protocol):
    def select(
        self,
        candidates: list[CandidateStructure],
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        """
        Selects the best candidates for labeling.

        Args:
            candidates: List of available candidates.
            potential_path: Path to the current potential (if any).
            work_dir: Directory for selection artifacts.

        Returns:
            A subset of candidates selected for labeling.
        """
        ...


class Oracle(Protocol):
    def compute(self, candidates: list[CandidateStructure], work_dir: Path) -> list[Path]:
        """
        Computes properties (energy, forces, etc.) for the candidate structures.

        Args:
            candidates: List of structures to compute.
            work_dir: Directory for computation artifacts.

        Returns:
            A list of paths to files containing the computed results (e.g., extxyz).
        """
        ...


class Trainer(Protocol):
    def train(self, dataset: Path, previous_potential: Path | None, output_dir: Path) -> Path:
        """
        Trains the potential using the provided dataset.

        Args:
            dataset: Path to the training dataset.
            previous_potential: Path to the previous potential (for fine-tuning).
            output_dir: Directory to save the trained potential.

        Returns:
            Path to the trained potential file.
        """
        ...

    def update_dataset(self, new_data_paths: list[Path]) -> Path:
        """
        Updates the dataset with new data.

        Args:
            new_data_paths: List of paths to new data files.

        Returns:
            Path to the updated dataset.
        """
        ...


class Validator(Protocol):
    def validate(self, potential_path: Path, work_dir: Path) -> ValidationResult:
        """
        Validates the potential.

        Args:
            potential_path: Path to the potential to validate.
            work_dir: Directory for validation artifacts.

        Returns:
            The result of the validation.
        """
        ...
