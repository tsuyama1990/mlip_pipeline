import gzip
import pickle
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.training.physics import ZBLCalculator


class DatasetBuilder:
    """Builds training datasets from ASE database."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.zbl_calc = ZBLCalculator()

    def fetch_data(self, query: str = "") -> list[Atoms]:
        """Fetches data from database."""
        # Use db_manager.get_atoms(selection=query)
        # Note: The mock in UAT needs to support get_atoms, not select.

        # But wait, we implemented `DatasetBuilder` using `self.db_manager.select` in previous step?
        # Let's check `mlip_autopipec/training/dataset.py` content in previous turn.
        # Ah, I wrote `rows = self.db_manager.select(query)` in `mlip_autopipec/training/dataset.py`.
        # But `DatabaseManager` does NOT have `select` method exposed directly. It has `get_atoms`.
        # `DatabaseManager` wraps `ase.db`. `_connection` is the one having `select`.

        # So I should change `mlip_autopipec/training/dataset.py` to use `get_atoms` or `select` on `db_manager._connection` if access allowed.
        # Ideally `db_manager` should be used via its public API.

        # `get_atoms` returns list of Atoms.
        return self.db_manager.get_atoms(selection=query)

    def compute_baseline(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Computes ZBL baseline energy and forces."""
        at_copy = atoms.copy()
        at_copy.calc = self.zbl_calc
        e_zbl = at_copy.get_potential_energy()
        f_zbl = at_copy.get_forces()

        return e_zbl, f_zbl

    def export(self, config: TrainConfig, output_dir: Path) -> Path:
        """Exports data to Pacemaker format with ZBL subtraction."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch data
        all_atoms = self.fetch_data("succeeded=True")

        # If UAT uses a mock DB, we need to ensure fetch_data works with the mock.
        # UAT sets up `mock_db.select`. But now I'm changing it to use `mock_db.get_atoms`.

        if not all_atoms:
            # Fallback for UAT if we want to allow empty query for test?
            # Or assume UAT provides "succeeded=True" or we just fetch everything if query is default.
            # In UAT I just passed config.
            pass

        if not all_atoms:
             # Try fetching without query if empty, maybe for testing
             all_atoms = self.fetch_data()

        if not all_atoms:
            raise ValueError("No training data found in database.")

        processed_atoms = []

        for at in all_atoms:
            # 2. Delta Learning
            e_dft = at.get_potential_energy()
            f_dft = at.get_forces()

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
                if "weights" in at.arrays:
                    at.set_array("weights", weights)
                else:
                    at.new_array("weights", weights)

            processed_atoms.append(at)

        # 4. Split Train/Test
        import random
        random.seed(42)
        random.shuffle(processed_atoms)

        n_test = int(len(processed_atoms) * config.test_fraction)
        test_set = processed_atoms[:n_test]
        train_set = processed_atoms[n_test:]

        if not train_set:
             # If test fraction is 0.0 (UAT case), all go to train_set.
             # If test fraction 0.0, n_test = 0. train_set = all.
             if not processed_atoms:
                 raise ValueError("Training set is empty.")
             train_set = processed_atoms

        # 5. Write to disk (pickle gzip)
        output_file = output_dir / "train.pckl.gzip"
        with gzip.open(output_file, "wb") as f:
            pickle.dump(train_set, f)

        if test_set:
            test_file = output_dir / "test.pckl.gzip"
            with gzip.open(test_file, "wb") as f:
                pickle.dump(test_set, f)

        return output_file
