from unittest.mock import patch
import numpy as np
from ase.atoms import Atoms
import yaml

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.training import TrainingConfig
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner


def test_uat_c04_01_full_training_loop(tmp_path):
    """
    UAT-C04-01: Full Training Loop
    """
    # Given a set of labelled structures
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    s1 = Structure.from_ase(atoms)
    s1.properties["energy"] = -5.0
    s1.properties["forces"] = np.zeros((2, 3))
    s1.properties["stress"] = np.zeros((3, 3))
    structures = [s1]

    # And a valid configuration
    config = TrainingConfig(batch_size=10, max_epochs=5)

    # When I run the training pipeline
    dataset_manager = DatasetManager()
    pacemaker = PacemakerRunner(work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # Mock pace_collect output
        dataset_path = tmp_path / "train.pckl.gzip"
        dataset_path.touch() # Side effect

        # Mock pace_train output
        (tmp_path / "potential.yace").touch()
        (tmp_path / "log.txt").write_text("Energy RMSE: 0.01 meV/atom\nForce RMSE: 0.01 eV/A")

        # 1. Convert
        data_file = dataset_manager.convert(structures, dataset_path)

        # Verify extxyz creation
        extxyz_path = tmp_path / "temp_dataset.extxyz"
        assert extxyz_path.exists()
        # Should contain Si atoms
        assert "Si" in extxyz_path.read_text()

        # 2. Train
        potential = pacemaker.train(config, data_file, elements=["Si"])

        # Verify input.yaml creation
        input_yaml = tmp_path / "input.yaml"
        assert input_yaml.exists()
        with open(input_yaml) as f:
            data = yaml.safe_load(f)
            assert data["backend"]["batch_size"] == 10

        # Then a potential should be created
        assert potential.path.exists()
        assert potential.path.name == "potential.yace"

        # And metrics should be available
        assert potential.metadata["rmse_energy"] == 0.01

def test_uat_c04_02_active_set_selection(tmp_path):
    """
    UAT-C04-02: Active Set Selection
    """
    # Given an existing dataset
    dataset_path = tmp_path / "train.pckl.gzip"
    dataset_path.touch()

    config = TrainingConfig(active_set_selection=True)
    pacemaker = PacemakerRunner(work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # Mock pace_activeset output
        active_dataset_path = tmp_path / "train_active.pckl.gzip"
        active_dataset_path.touch()

        # When I select active set
        result_path = pacemaker.select_active_set(config, dataset_path)

        # Then the reduced dataset is returned
        assert result_path == active_dataset_path

        # And pace_activeset was called
        args = mock_run.call_args[0][0]
        assert "pace_activeset" in args
        assert str(dataset_path) in args
