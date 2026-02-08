from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.trainer.pacemaker import PacemakerTrainer
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.config import PacemakerTrainerConfig
from mlip_autopipec.domain_models.structure import Structure


def create_dummy_structure():
    return Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        energy=-1.5,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )

@pytest.fixture
def mock_dataset(tmp_path):
    dataset = Dataset(tmp_path / "dataset.jsonl", root_dir=tmp_path)
    # Add dummy data
    dataset.append([create_dummy_structure()])
    return dataset

@pytest.fixture
def trainer_config():
    return PacemakerTrainerConfig(
        name="pacemaker",
        basis_size=10,
        cutoff=5.0,
        active_set_selection=True,
        active_set_limit=100
    )

def test_pacemaker_trainer_execution(tmp_path, trainer_config, mock_dataset):
    # Mock ActiveSetSelector
    with patch("mlip_autopipec.components.trainer.pacemaker.ActiveSetSelector") as MockSelector:
        mock_selector_instance = MockSelector.return_value
        mock_selector_instance.select.return_value = tmp_path / "dataset_activeset.pckl.gzip"

        # Mock subprocess.run for pace_train
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create dummy potential file that trainer expects
            (tmp_path / "potential.yace").touch()

            trainer = PacemakerTrainer(trainer_config)

            potential = trainer.train(mock_dataset, tmp_path)

            # Verify potential object
            assert potential.path == tmp_path / "potential.yace"

            # Verify Active Set was called
            mock_selector_instance.select.assert_called_once()

            # Verify pace_train was called
            args, _ = mock_run.call_args
            command = args[0]
            assert "pace_train" in command
            assert "input.yaml" in str(command)

def test_pacemaker_trainer_config_generation(tmp_path, trainer_config, mock_dataset):
    # This test verifies that input.yaml is generated correctly
    with patch("mlip_autopipec.components.trainer.pacemaker.ActiveSetSelector") as MockSelector, \
         patch("subprocess.run") as mock_run:

        mock_selector_instance = MockSelector.return_value
        mock_selector_instance.select.return_value = tmp_path / "dataset_activeset.pckl.gzip"
        mock_run.return_value = MagicMock(returncode=0)
        (tmp_path / "potential.yace").touch()

        trainer = PacemakerTrainer(trainer_config)
        trainer.train(mock_dataset, tmp_path)

        input_yaml = tmp_path / "input.yaml"
        assert input_yaml.exists()

        # Check content
        import yaml
        with input_yaml.open() as f:
            config = yaml.safe_load(f)

        assert config["cutoff"] == 5.0
        assert config["data"]["filename"] == "dataset_activeset.pckl.gzip"
