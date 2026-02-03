import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.config import TrainingConfig
from mlip_autopipec.domain_models.structures import CandidateStructure
from mlip_autopipec.orchestration.interfaces import Trainer

logger = logging.getLogger(__name__)


class PacemakerTrainer(Trainer):
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, dataset: Path, previous_potential: Path | None, output_dir: Path) -> Path:
        """
        Runs pace_train via subprocess or mock if environment variable PYACEMAKER_MOCK_MODE is set.
        """
        mock_mode = os.environ.get("PYACEMAKER_MOCK_MODE", "0") == "1"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # In Spec example, output filename is "output_potential.yace" inside output_dir
        expected_output = output_dir / "output_potential.yace"

        if mock_mode:
            logger.info("Running PacemakerTrainer in MOCK MODE")
            expected_output.touch()
            return expected_output

        cmd = [self.config.command, "--dataset", str(dataset)]
        if previous_potential:
            cmd.extend(["--initial_potential", str(previous_potential)])

        cmd.extend(["--max_num_epochs", str(self.config.max_epochs)])
        cmd.extend(["--output_dir", str(output_dir)])

        logger.info(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")  # noqa: TRY400
            raise

        # Verify output exists
        if not expected_output.exists():
            msg = f"Expected output file {expected_output} was not created."
            logger.error(msg)
            raise FileNotFoundError(msg)

        return expected_output

    def update_dataset(self, new_data_paths: list[Path]) -> Path:
        """
        Updates the dataset with new data by appending new structures.
        """
        if not new_data_paths:
            return self.config.dataset_path

        logger.info(
            f"Updating dataset {self.config.dataset_path} with {len(new_data_paths)} new files."
        )

        # Read existing dataset if it exists
        all_atoms: list[Any] = []
        if self.config.dataset_path.exists():
             # Use a generic format or allow config to specify. Defaulting to extxyz/pckl support via ASE
            try:
                read_data = read(str(self.config.dataset_path), index=":")
                if isinstance(read_data, list):
                    all_atoms = read_data
                else:
                    all_atoms = [read_data]
            except Exception:
                logger.warning(f"Could not read existing dataset at {self.config.dataset_path}. Starting fresh.")
                all_atoms = []

        # Read and append new data
        for path in new_data_paths:
            try:
                # Use index=":" to ensure we read all structures, not just the last one
                atoms = read(str(path), index=":")
                if isinstance(atoms, list):
                    all_atoms.extend(atoms)
                else:
                    all_atoms.append(atoms)
            except Exception:
                logger.exception(f"Failed to read data from {path}")

        # Write back
        # Ensure parent dir exists
        self.config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        # Using extxyz as it preserves info better than basic xyz, or whatever extension is provided
        write(str(self.config.dataset_path), all_atoms)

        return self.config.dataset_path

    def select_candidates(
        self, candidates: list[CandidateStructure], n_selection: int
    ) -> list[CandidateStructure]:
        """
        Selects a subset of candidates.
        """
        mock_mode = os.environ.get("PYACEMAKER_MOCK_MODE", "0") == "1"

        if mock_mode:
            logger.info(f"Selecting {n_selection} candidates (MOCK MODE: Slicing)")
            return candidates[:n_selection]

        # Real mode implementation:
        # Ideally, we would run `pace_activeset` command here.
        # Since we don't have the full CLI wrapper for activeset yet, we will fallback to slicing
        # with a warning, or if we want to be strict, we could raise NotImplementedError.
        # However, to allow "skeleton loop" to work if not in mock mode but without external deps:
        logger.warning("pace_activeset integration not yet implemented. Falling back to simple slicing.")
        return candidates[:n_selection]
