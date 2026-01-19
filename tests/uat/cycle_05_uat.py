"""
User Acceptance Test (UAT) for Cycle 05: Active Learning & Training.

This script validates the key requirements:
1. Dataset export from database with correct ZBL baseline subtraction.
2. Respect for force masking (weights) in training data.
3. Correct generation of Pacemaker configuration files (input.yaml).
4. End-to-end execution of the Pacemaker training wrapper (mocked subprocess).
"""

import gzip
import pickle
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.schemas.training import TrainConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.training.config_gen import TrainConfigGenerator
from mlip_autopipec.training.dataset import DatasetBuilder
from mlip_autopipec.training.pacemaker import PacemakerWrapper

# ANSI Colors for Output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def run_uat():
    work_dir = Path("uat_cycle_05_workspace")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    try:
        # ---------------------------------------------------------------------
        # Scenario 1: Dataset Export & Delta Learning
        # Goal: Verify that ZBL energy is correctly subtracted from DFT energy.
        # ---------------------------------------------------------------------

        # Setup mock DB with synthetic data: H-H at 0.5A (repulsive).
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.5]])
        # Set a high DFT Energy = 200.0 eV
        atoms.calc = SinglePointCalculator(atoms, energy=200.0, forces=np.zeros((2, 3)))

        mock_db = Mock(spec=DatabaseManager)
        mock_db.get_atoms.return_value = [atoms]

        dataset_builder = DatasetBuilder(mock_db)
        config = TrainConfig(enable_delta_learning=True, test_fraction=0.0)

        data_path = dataset_builder.export(config, work_dir)

        with gzip.open(data_path, "rb") as f:
            atoms_list = pickle.load(f)

        at = atoms_list[0]

        # Verify ZBL Subtraction: New E = DFT E - ZBL E
        delta_e = at.info["energy"]
        zbl_e = at.info["zbl_energy"]

        if abs(delta_e - (200.0 - zbl_e)) < 1e-5:
            pass
        else:
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Scenario 2: Masking Respect
        # Goal: Verify that atoms with `force_mask` array get a corresponding `weights` array.
        # ---------------------------------------------------------------------

        at_masked = atoms.copy()
        at_masked.set_array("force_mask", np.array([1, 0]))
        at_masked.calc = SinglePointCalculator(at_masked, energy=200.0, forces=np.zeros((2, 3)))

        mock_db.get_atoms.return_value = [at_masked]

        data_path = dataset_builder.export(config, work_dir)

        with gzip.open(data_path, "rb") as f:
            atoms_list = pickle.load(f)

        at = atoms_list[0]
        if "weights" in at.arrays and np.array_equal(at.arrays["weights"], np.array([1.0, 0.0])):
            pass
        else:
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Scenario 3: Pacemaker Configuration
        # Goal: Verify that Jinja2 templates render the correct values from TrainConfig.
        # ---------------------------------------------------------------------

        template_dir = work_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        template_path = template_dir / "input.yaml.j2"

        template_content = """
cutoff: {{ config.cutoff }}
data_path: {{ data_path }}
loss:
    kappa_f: {{ config.loss_weights.forces }}
"""
        with open(template_path, "w") as f:
            f.write(template_content)

        config_gen = TrainConfigGenerator(template_path)
        config = TrainConfig(
            cutoff=5.5, loss_weights={"energy": 1.0, "forces": 50.0, "stress": 10.0}
        )

        dummy_data = work_dir / "train.pckl.gzip"
        output_pot = work_dir / "output.yace"

        input_yaml = config_gen.generate(config, dummy_data, output_pot, elements=["H"])

        with open(input_yaml) as f:
            yaml_content = yaml.safe_load(f)

        if yaml_content["cutoff"] == 5.5 and yaml_content["loss"]["kappa_f"] == 50.0:
            pass
        else:
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Scenario 4: Training Execution & Monitoring
        # Goal: Verify the PacemakerWrapper runs the full flow and parses logs correctly.
        # ---------------------------------------------------------------------

        wrapper = PacemakerWrapper(executable="pacemaker")

        # Mock the subprocess call since Pacemaker is not installed
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = """
            ...
            RMSE (energy) : 0.005
            RMSE (forces) : 0.123
            ...
            """
            mock_run.return_value.returncode = 0

            # Create dummy output file to simulate successful training
            output_pot.touch()

            result = wrapper.train(config, dataset_builder, config_gen, work_dir, generation=1)

            if (
                result.rmse_energy == 0.005
                and result.rmse_forces == 0.123
                and result.generation == 1
            ):
                pass
            else:
                sys.exit(1)

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    run_uat()
