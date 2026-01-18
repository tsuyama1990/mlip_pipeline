"""
Module for preparing training datasets for Pacemaker.
Handles database querying, data validation, delta learning (ZBL), and file export.
"""

import gzip
import logging
import pickle
import random
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.data_models.training_data import TrainingBatch
from mlip_autopipec.training.physics import ZBLCalculator

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds training datasets from ASE database for Pacemaker training.

    Attributes:
        db_manager: Interface to the ASE database.
        zbl_calc: Calculator for ZBL baseline energy/forces.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the DatasetBuilder.

        Args:
            db_manager: An initialized DatabaseManager instance.
        """
        self.db_manager = db_manager
        self.zbl_calc = ZBLCalculator()

    def fetch_data(self, query: str = "") -> list[Atoms]:
        """
        Fetches data from the database and validates it using Pydantic.

        Args:
            query: ASE-db selection query string (e.g., 'succeeded=True').

        Returns:
            A list of valid ase.Atoms objects.

        Raises:
            ValidationError: If the fetched data does not conform to the schema.
        """
        logger.info(f"Fetching training data with query: '{query}'")
        atoms_list = self.db_manager.get_atoms(selection=query)

        # Validate input boundaries using Pydantic
        # TrainingBatch validates that it is a list of ase.Atoms
        batch = TrainingBatch(atoms_list=atoms_list)

        logger.info(f"Fetched and validated {len(batch.atoms_list)} structures.")
        return batch.atoms_list

    def compute_baseline(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """
        Computes ZBL baseline energy and forces for a structure.

        Args:
            atoms: The atomic structure.

        Returns:
            Tuple of (ZBL Energy [eV], ZBL Forces [eV/A]).
        """
        # Create a copy to avoid attaching calculator to the original object permanently
        at_copy = atoms.copy()
        at_copy.calc = self.zbl_calc
        e_zbl = at_copy.get_potential_energy()
        f_zbl = at_copy.get_forces()

        return e_zbl, f_zbl

    def _fetch_all_atoms(self) -> list[Atoms]:
        """Fetches all relevant atoms from the database."""
        # Try fetching successfully converged calculations
        query = "succeeded=True"
        all_atoms = self.fetch_data(query)

        # Fallback logic for testing environments where 'succeeded' might not be set
        if not all_atoms:
            logger.warning(f"No atoms found with query '{query}'. Attempting to fetch all atoms.")
            all_atoms = self.fetch_data("")

        if not all_atoms:
            logger.error("No training data found in database.")
            raise ValueError("No training data found in database.")

        return all_atoms

    def _process_atom(self, at: Atoms, config: TrainConfig) -> Atoms | None:
        """
        Processes a single atom: checks data, applies Delta Learning and Force Masking.
        Returns the processed atom or None if it should be skipped.
        """
        # Check for required properties
        try:
            e_dft = at.get_potential_energy()
            f_dft = at.get_forces()
        except RuntimeError:
            logger.warning(f"Skipping atom {at} - missing energy or forces.")
            return None

        # Delta Learning
        if config.enable_delta_learning:
            e_zbl, f_zbl = self.compute_baseline(at)
            at.info["energy"] = e_dft - e_zbl
            at.arrays["forces"] = f_dft - f_zbl
            at.info["zbl_energy"] = e_zbl
            at.arrays["zbl_forces"] = f_zbl
        else:
            at.info["energy"] = e_dft
            at.arrays["forces"] = f_dft

        # Handle Force Masking
        if "force_mask" in at.arrays:
            mask = at.arrays["force_mask"]
            weights = mask.astype(float)
            # Pacemaker/ACE usually looks for 'weights' array
            if "weights" in at.arrays:
                at.set_array("weights", weights)
            else:
                at.new_array("weights", weights)

        return at

    def _split_dataset(self, processed_atoms: list[Atoms], config: TrainConfig) -> tuple[list[Atoms], list[Atoms]]:
        """Splits the dataset into training and testing sets."""
        random.seed(42)
        random.shuffle(processed_atoms)

        n_test = int(len(processed_atoms) * config.test_fraction)
        test_set = processed_atoms[:n_test]
        train_set = processed_atoms[n_test:]

        if not train_set:
             logger.error("Training set is empty after splitting.")
             raise ValueError("Training set is empty.")

        return train_set, test_set

    def _write_dataset(self, train_set: list[Atoms], test_set: list[Atoms], output_dir: Path) -> Path:
        """Writes the training and testing sets to gzip-pickled files."""
        output_file = output_dir / "train.pckl.gzip"
        logger.info(f"Writing {len(train_set)} structures to {output_file}")
        with gzip.open(output_file, "wb") as f:
            pickle.dump(train_set, f)

        if test_set:
            test_file = output_dir / "test.pckl.gzip"
            logger.info(f"Writing {len(test_set)} structures to {test_file}")
            with gzip.open(test_file, "wb") as f:
                pickle.dump(test_set, f)

        return output_file

    def export(self, config: TrainConfig, output_dir: Path) -> Path:
        """
        Exports data to Pacemaker format with optional ZBL subtraction.

        Args:
            config: Training configuration.
            output_dir: Directory to save the exported files.

        Returns:
            Path to the main training dataset file.

        Raises:
            ValueError: If no data is found or split results in empty training set.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch data
        all_atoms = self._fetch_all_atoms()

        # 2. Process atoms
        processed_atoms = []
        logger.info("Processing atoms (Delta Learning & Masking)...")
        for at in all_atoms:
            processed_at = self._process_atom(at, config)
            if processed_at:
                processed_atoms.append(processed_at)

        logger.info(f"Processed {len(processed_atoms)} valid atoms.")

        if not processed_atoms:
             logger.error("No valid atoms remaining after processing.")
             raise ValueError("No valid atoms found after processing.")

        # 3. Split Train/Test
        train_set, test_set = self._split_dataset(processed_atoms, config)

        # 4. Write to disk
        return self._write_dataset(train_set, test_set, output_dir)
