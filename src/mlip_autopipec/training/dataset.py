"""
Module for preparing training datasets for Pacemaker.
Handles database querying and file export.
"""

import logging
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.utils.config_utils import validate_path_safety

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VALIDATION_RATIO = 0.1


class DatasetBuilder:
    """
    Builds training datasets from ASE database for Pacemaker training.

    This class isolates the data access and transformation logic required
    to prepare the ExtXYZ files for Pacemaker. It decouples the raw
    database format from the training engine requirements.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """
        Initialize the DatasetBuilder.

        Args:
            db_manager: An initialized DatabaseManager instance.
        """
        self.db_manager = db_manager

    def export_atoms(self, atoms_iterable: Iterable[Atoms], output_path: Path) -> None:
        """
        Exports a collection of atoms to ExtXYZ format.

        Args:
            atoms_iterable: Iterable of ASE Atoms objects.
            output_path: Destination file path.
        """
        output_path = validate_path_safety(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w") as f:
                for atoms in atoms_iterable:
                    write(f, atoms, format="extxyz")
        except Exception as e:
            logger.exception(f"Failed to export atoms to {output_path}: {e}")
            raise

    def export(self, config: TrainingConfig, output_dir: Path) -> Path:
        """
        Exports data to Pacemaker format (ExtXYZ) using streaming to avoid OOM.

        This method queries the database for completed structures and splits them
        probabilistically into training and validation sets. It keeps file handles open
        for efficiency.

        Args:
            config: Training configuration containing file paths and parameters.
            output_dir: Base directory where data files will be saved.

        Returns:
            Path to the training data file.

        Raises:
            ValueError: If no data is found in the database.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate paths
        train_path = validate_path_safety(output_dir / config.training_data_path)
        test_path = validate_path_safety(output_dir / config.test_data_path)

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        query = "status=completed"
        logger.info(f"Fetching training data with query: '{query}'")

        rng = random.Random(42)
        count = 0
        train_count = 0
        test_count = 0

        try:
            # Use 'w' mode to start fresh
            with train_path.open("w") as f_train, test_path.open("w") as f_test:
                # Streaming from DB generator
                for atoms in self.db_manager.select(selection=query):
                    # Fix stress if it is a list (retrieved from JSON data)
                    if "stress" in atoms.info and isinstance(atoms.info["stress"], list):
                        atoms.info["stress"] = np.array(atoms.info["stress"])

                    count += 1
                    if rng.random() < DEFAULT_VALIDATION_RATIO:
                        write(f_test, atoms, format="extxyz")
                        test_count += 1
                    else:
                        write(f_train, atoms, format="extxyz")
                        train_count += 1

        except Exception as e:
            logger.exception(f"Error during data export: {e}")
            raise

        if count == 0:
            logger.error(f"No atoms found with query '{query}'")
            msg = "No training data found in database."
            raise ValueError(msg)

        if test_count == 0 and train_count > 0:
            logger.warning("Validation set empty after split.")

        logger.info(
            f"Exported {count} structures: {train_count} training, {test_count} validation."
        )
        return train_path
