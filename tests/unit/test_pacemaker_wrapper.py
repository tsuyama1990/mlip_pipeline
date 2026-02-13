from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.enums import ActiveSetMethod, TrainerType
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.trainer.pacemaker_wrapper import PacemakerTrainer


@pytest.fixture
def trainer_config() -> TrainerConfig:
    return TrainerConfig(
        type=TrainerType.PACEMAKER,
        cutoff=5.0,
        order=2,
        basis_size=100,
        delta_learning="zbl",
        max_epochs=10,
        seed=12345,
        kappa=0.5
    )


@pytest.fixture
def sample_structures() -> list[Structure]:
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    s1 = Structure(atoms=atoms, provenance="test", label_status="labeled", energy=-10.0, forces=[[0,0,0]]*3, stress=[0]*6)
    return [s1]


@patch("mlip_autopipec.trainer.pacemaker_wrapper.DatasetManager")
@patch("subprocess.run")
def test_train_flow(
    mock_run: MagicMock,
    mock_dataset_manager_cls: MagicMock,
    trainer_config: TrainerConfig,
    sample_structures: list[Structure],
    tmp_path: Path
) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    # Mock DatasetManager instance methods
    mock_dm_instance = mock_dataset_manager_cls.return_value
    # Set return value to tuple: path, elements, count
    mock_dm_instance.create_dataset.return_value = (tmp_path / "dataset.pckl.gzip", ["H", "O"], 1)
    mock_dm_instance.select_active_set.return_value = tmp_path / "dataset_active.pckl.gzip"

    # Mock subprocess return code for pace_train
    mock_run.return_value.returncode = 0

    # Mock potential file creation (fake it)
    (tmp_path / "output_potential.yace").touch()

    potential = trainer.train(sample_structures)

    # Verify DatasetManager calls
    mock_dm_instance.create_dataset.assert_called_once()

    assert mock_run.called
    args = mock_run.call_args[0][0]
    # Check command name
    assert args[0] == "pace_train"
    # Check input file passed
    assert "input.yaml" in str(args[1])

    assert isinstance(potential, Potential)
    assert potential.format == "yace"
    assert potential.path == tmp_path / "output_potential.yace"


@patch("mlip_autopipec.trainer.pacemaker_wrapper.DatasetManager")
@patch("subprocess.run")
def test_train_config_generation(
    mock_run: MagicMock,
    mock_dataset_manager_cls: MagicMock,
    trainer_config: TrainerConfig,
    sample_structures: list[Structure],
    tmp_path: Path
) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    mock_dm_instance = mock_dataset_manager_cls.return_value
    mock_dm_instance.create_dataset.return_value = (tmp_path / "dataset.pckl.gzip", ["H", "O"], 1)
    mock_run.return_value.returncode = 0
    (tmp_path / "output_potential.yace").touch()

    trainer.train(sample_structures)

    input_yaml_path = tmp_path / "input.yaml"
    assert input_yaml_path.exists()
    content = input_yaml_path.read_text()

    assert "cutoff: 5.0" in content
    assert "seed: 12345" in content
    assert "kappa: 0.5" in content
    # Verify delta learning section
    assert "pair_style: zbl" in content


@patch("mlip_autopipec.trainer.pacemaker_wrapper.DatasetManager")
def test_train_empty_structures(mock_dataset_manager_cls: MagicMock, trainer_config: TrainerConfig, tmp_path: Path) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    mock_dm_instance = mock_dataset_manager_cls.return_value
    # Mock create_dataset returning 0 count
    mock_dm_instance.create_dataset.return_value = (tmp_path / "dataset.pckl.gzip", [], 0)

    with pytest.raises(ValueError, match="No structures provided"):
        trainer.train([])


@patch("mlip_autopipec.trainer.pacemaker_wrapper.DatasetManager")
@patch("subprocess.run")
def test_train_with_active_set(
    mock_run: MagicMock,
    mock_dataset_manager_cls: MagicMock,
    trainer_config: TrainerConfig,
    sample_structures: list[Structure],
    tmp_path: Path
) -> None:
    # Enable active set
    trainer_config.active_set_method = ActiveSetMethod.MAXVOL
    trainer_config.selection_ratio = 0.5

    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    mock_dm_instance = mock_dataset_manager_cls.return_value
    # Return count=10 so selection_ratio=0.5 selects 5
    mock_dm_instance.create_dataset.return_value = (tmp_path / "dataset.pckl.gzip", ["H"], 10)
    mock_dm_instance.select_active_set.return_value = tmp_path / "dataset_active.pckl.gzip"
    mock_run.return_value.returncode = 0
    (tmp_path / "output_potential.yace").touch()

    trainer.train(sample_structures) # The actual list passed doesn't matter as we mocked create_dataset return

    mock_dm_instance.select_active_set.assert_called_once()
    args = mock_dm_instance.select_active_set.call_args
    # Check path and count
    assert args[0][1] == 5

    # Check if input.yaml uses the active dataset
    input_yaml_path = tmp_path / "input.yaml"
    content = input_yaml_path.read_text()
    assert "dataset_active.pckl.gzip" in content


@patch("mlip_autopipec.trainer.pacemaker_wrapper.DatasetManager")
@patch("subprocess.run")
def test_train_failure(
    mock_run: MagicMock,
    mock_dataset_manager_cls: MagicMock,
    trainer_config: TrainerConfig,
    sample_structures: list[Structure],
    tmp_path: Path
) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "Error"

    mock_dm_instance = mock_dataset_manager_cls.return_value
    mock_dm_instance.create_dataset.return_value = (tmp_path / "dataset.pckl.gzip", ["H"], 1)

    with pytest.raises(RuntimeError, match="pace_train failed"):
        trainer.train(sample_structures)


@patch("mlip_autopipec.trainer.pacemaker_wrapper.DatasetManager")
@patch("subprocess.run")
def test_train_missing_output(
    mock_run: MagicMock,
    mock_dataset_manager_cls: MagicMock,
    trainer_config: TrainerConfig,
    sample_structures: list[Structure],
    tmp_path: Path
) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    mock_dm_instance = mock_dataset_manager_cls.return_value
    mock_dm_instance.create_dataset.return_value = (tmp_path / "dataset.pckl.gzip", ["H"], 1)

    mock_run.return_value.returncode = 0
    # Ensure no .yace files exist
    for f in tmp_path.glob("*.yace"):
        f.unlink()

    with pytest.raises(FileNotFoundError, match="did not produce a .yace file"):
        trainer.train(sample_structures)
