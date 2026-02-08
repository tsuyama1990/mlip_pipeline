from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.components.trainer.pacemaker import PacemakerTrainer
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.config import PacemakerTrainerConfig
from mlip_autopipec.domain_models.enums import TrainerType
from mlip_autopipec.domain_models.structure import Structure


def create_dummy_structure() -> Structure:
    return Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        energy=-1.5,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3)),
    )


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Dataset:
    dataset = Dataset(tmp_path / "dataset.jsonl", root_dir=tmp_path)
    # Add dummy data
    dataset.append([create_dummy_structure()])
    return dataset


@pytest.fixture
def trainer_config() -> PacemakerTrainerConfig:
    return PacemakerTrainerConfig(
        name=TrainerType.PACEMAKER,
        basis_size=10,
        cutoff=5.0,
        active_set_selection=True,
        active_set_limit=100,
    )


def test_pacemaker_trainer_execution(
    tmp_path: Path, trainer_config: PacemakerTrainerConfig, mock_dataset: Dataset
) -> None:
    # Mock ActiveSetSelector
    with patch("mlip_autopipec.components.trainer.pacemaker.ActiveSetSelector") as MockSelector:
        mock_selector_instance = MockSelector.return_value

        # Must verify active set selector returns absolute path to existing file
        activeset_path = tmp_path / trainer_config.activeset_filename
        activeset_path.touch()
        mock_selector_instance.select.return_value = activeset_path

        # Mock subprocess.run for pace_train
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create dummy potential file that trainer expects
            (tmp_path / trainer_config.potential_filename).touch()

            trainer = PacemakerTrainer(trainer_config)

            potential = trainer.train(mock_dataset, tmp_path)

            # Verify potential object
            assert potential.path == tmp_path / trainer_config.potential_filename

            # Verify Active Set was called
            mock_selector_instance.select.assert_called_once()

            # Verify pace_train was called
            args, _ = mock_run.call_args
            command = args[0]
            assert "pace_train" in command

            # Verify input.yaml path in command is absolute
            input_yaml_arg = command[1]
            assert str(tmp_path / trainer_config.input_filename) == input_yaml_arg

            # Verify input.yaml exists
            assert (tmp_path / trainer_config.input_filename).exists()

            # Verify command arguments structure
            # args[0] is the command list passed to subprocess.run
            # e.g. ['pace_train', '/path/to/input.yaml']
            assert len(command) == 2
            assert command[0] == "pace_train"
            assert command[1] == str(tmp_path / trainer_config.input_filename)

            # Verify dataset.extxyz exists (default format)
            # Default is extxyz now in Config? No, default is extxyz
            if trainer_config.data_format == "extxyz":
                extxyz_path = (
                    tmp_path / Path(trainer_config.dataset_filename).with_suffix(".extxyz").name
                )
                assert extxyz_path.exists()


def test_pacemaker_trainer_config_generation(
    tmp_path: Path, trainer_config: PacemakerTrainerConfig, mock_dataset: Dataset
) -> None:
    # This test verifies that input.yaml is generated correctly
    with (
        patch("mlip_autopipec.components.trainer.pacemaker.ActiveSetSelector") as MockSelector,
        patch("subprocess.run") as mock_run,
    ):
        mock_selector_instance = MockSelector.return_value
        activeset_path = tmp_path / trainer_config.activeset_filename
        activeset_path.touch()
        mock_selector_instance.select.return_value = activeset_path

        mock_run.return_value = MagicMock(returncode=0)
        (tmp_path / trainer_config.potential_filename).touch()

        trainer = PacemakerTrainer(trainer_config)
        trainer.train(mock_dataset, tmp_path)

        input_yaml = tmp_path / trainer_config.input_filename
        assert input_yaml.exists()

        # Check content
        import yaml

        with input_yaml.open() as f:
            config = yaml.safe_load(f)

        assert config["cutoff"] == 5.0
        # Should point to active set file name relative to workdir, or absolute?
        # In implementation: config_dict["data"]["filename"] = data_path.name (relative)
        # Because we run pace_train with cwd=workdir
        assert config["data"]["filename"] == trainer_config.activeset_filename
