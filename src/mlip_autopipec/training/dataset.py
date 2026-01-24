"""
Module for preparing training datasets for Pacemaker.
Handles database querying and file export.
"""

import logging
import random
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.orchestration.database import DatabaseManager

logger = logging.getLogger(__name__)

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

    def export_atoms(self, atoms_list: list[Atoms], output_path: Path) -> None:
        """
        Exports a list of atoms to ExtXYZ format.

        Args:
            atoms_list: List of ASE Atoms objects.
            output_path: Destination file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write(str(output_path), atoms_list, format="extxyz")
        except Exception as e:
            logger.error(f"Failed to export atoms to {output_path}: {e}")
            raise

    def export(self, config: TrainingConfig, output_dir: Path) -> Path:
        """
        Exports data to Pacemaker format (ExtXYZ).

        This method queries the database for completed structures, splits them
        into training and validation sets, and writes them to the paths specified
        in the configuration relative to the output directory.

        Args:
            config: Training configuration containing file paths and parameters.
            output_dir: Base directory where data files will be saved.

        Returns:
            Path to the training data file.

        Raises:
            ValueError: If no data is found in the database.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch data
        query = "status=completed"
        logger.info(f"Fetching training data with query: '{query}'")
        atoms_list = self.db_manager.get_atoms(selection=query)

        if not atoms_list:
            logger.error(f"No atoms found with query '{query}'")
            raise ValueError("No training data found in database.")

        logger.info(f"Fetched {len(atoms_list)} structures.")

        # 2. Split Train/Test (90/10)
        # Using fixed seed for reproducibility as per SPEC
        random.seed(42)
        random.shuffle(atoms_list)

        n_total = len(atoms_list)
        # Fixed 10% validation as per SPEC Summary
        n_test = max(1, int(n_total * 0.1))
        # Ensure we have at least 1 train if possible, but if n_total=1, n_test=1, train=0.
        # Handle edge case for very small datasets (e.g. testing)
        if n_total < 2:
            n_test = 0 # If only 1 atom, use it for training? Or fail?
                       # SPEC says "Bridge gap... populated". Assuming reasonable size.
                       # But for robustness:
            if n_total > 0:
                 # If we have data, prioritize training
                 n_test = 0

        test_set = atoms_list[:n_test]
        train_set = atoms_list[n_test:]

        if not train_set and n_total > 0:
             # Fallback if math weirdness happened (shouldn't with above logic)
             train_set = atoms_list
             test_set = []

        logger.info(f"Split data: {len(train_set)} training, {len(test_set)} validation.")

        # 3. Write files
        train_path = output_dir / config.training_data_path
        test_path = output_dir / config.test_data_path

        logger.info(f"Writing {len(train_set)} structures to {train_path}")
        self.export_atoms(train_set, train_path)

        if test_set:
            logger.info(f"Writing {len(test_set)} structures to {test_path}")
            self.export_atoms(test_set, test_path)
        else:
            # Create empty file or warn? SPEC implies both exist.
            # If no test data, maybe copy train? Or just leave empty file?
            # Pacemaker might fail if test file defined but missing.
            # UAT says "Files exist".
            logger.warning("Validation set is empty. Creating empty test file.")
            test_path.touch()

        return train_path
