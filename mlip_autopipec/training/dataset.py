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

    def export(self, config: TrainConfig, output_dir: Path) -> Path:
        """
        Exports data to Pacemaker format with optional ZBL subtraction.

        Steps:
        1. Fetch data from DB.
        2. Validate.
        3. Subtract ZBL baseline (if enabled).
        4. Apply force masks (weights).
        5. Split into Train/Test sets.
        6. Write to gzip-pickled files.

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

        processed_atoms = []

        logger.info("Processing atoms (Delta Learning & Masking)...")
        for at in all_atoms:
            # Check for required properties
            # get_potential_energy() might raise RuntimeError if no calculator/results attached
            try:
                e_dft = at.get_potential_energy()
                f_dft = at.get_forces()
            except RuntimeError:
                logger.warning(f"Skipping atom {at} - missing energy or forces.")
                continue

            # 2. Delta Learning
            if config.enable_delta_learning:
                e_zbl, f_zbl = self.compute_baseline(at)
                at.info["energy"] = e_dft - e_zbl
                at.arrays["forces"] = f_dft - f_zbl
                at.info["zbl_energy"] = e_zbl
                at.arrays["zbl_forces"] = f_zbl
            else:
                at.info["energy"] = e_dft
                at.arrays["forces"] = f_dft

            # 3. Handle Force Masking
            if "force_mask" in at.arrays:
                mask = at.arrays["force_mask"]
                weights = mask.astype(float)
                # Pacemaker/ACE usually looks for 'weights' array
                if "weights" in at.arrays:
                    at.set_array("weights", weights)
                else:
                    at.new_array("weights", weights)

            processed_atoms.append(at)

        logger.info(f"Processed {len(processed_atoms)} valid atoms.")

        # 4. Split Train/Test
        random.seed(42)
        random.shuffle(processed_atoms)

        n_test = int(len(processed_atoms) * config.test_fraction)
        test_set = processed_atoms[:n_test]
        train_set = processed_atoms[n_test:]

        # Handle case where test_fraction=0.0 (e.g. UAT or specific training modes)
        if not train_set and processed_atoms:
             # If processed_atoms is not empty but train_set is, it means n_test took everything?
             # If test_fraction >= 1.0 this happens. Schema restricts test_fraction < 1.0.
             # So this is unlikely unless processed_atoms is empty, handled above.
             # But if processed_atoms has 1 item and test_fraction is 0.9, n_test=0. 1 goes to train.
             pass

        if not train_set:
             logger.error("Training set is empty after splitting.")
             raise ValueError("Training set is empty.")

        # 5. Write to disk
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
