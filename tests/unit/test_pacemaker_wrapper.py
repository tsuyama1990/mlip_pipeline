from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.enums import TrainerType
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
        max_epochs=10
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
    mock_dm_instance.create_dataset.return_value = tmp_path / "dataset.pckl.gzip"
    mock_dm_instance.select_active_set.return_value = tmp_path / "dataset_active.pckl.gzip"

    # Mock subprocess return code for pace_train
    mock_run.return_value.returncode = 0

    # Mock potential file creation (fake it)
    (tmp_path / "potential.yace").touch()

    potential = trainer.train(sample_structures)

    # Verify DatasetManager calls
    mock_dm_instance.create_dataset.assert_called_once()

    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[0] == "pace_train"
    assert "input.yaml" in str(args)

    assert isinstance(potential, Potential)
    assert potential.format == "yace"
    assert potential.path == tmp_path / "potential.yace"

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

    # To test config generation, we can check the file written.
    # But subprocess run is mocked, so maybe the file is written before run.
    # We can inspect the file if we let it write.

    mock_dm_instance = mock_dataset_manager_cls.return_value
    mock_dm_instance.create_dataset.return_value = tmp_path / "dataset.pckl.gzip"
    mock_dm_instance.select_active_set.return_value = tmp_path / "dataset.pckl.gzip"
    mock_run.return_value.returncode = 0
    (tmp_path / "potential.yace").touch()

    trainer.train(sample_structures)

    input_yaml_path = tmp_path / "input.yaml"
    assert input_yaml_path.exists()
    content = input_yaml_path.read_text()

    assert "cutoff: 5.0" in content
    assert "max_deg: 2" in content # or similar key for order
    # Verify delta learning section
    assert "potential: zbl" in content or "pair_style: zbl" in content

def test_train_empty_structures(trainer_config: TrainerConfig, tmp_path: Path) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)
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
    from mlip_autopipec.domain_models.enums import ActiveSetMethod
    trainer_config.active_set_method = ActiveSetMethod.MAXVOL
    trainer_config.selection_ratio = 0.5 # Should select 0 if len=1? int(1*0.5) = 0.
    # Need more structures to get count > 0
    structures = sample_structures * 2 # len=2, count=1

    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    mock_dm_instance = mock_dataset_manager_cls.return_value
    mock_dm_instance.create_dataset.return_value = tmp_path / "dataset.pckl.gzip"
    mock_dm_instance.select_active_set.return_value = tmp_path / "dataset_active.pckl.gzip"
    mock_run.return_value.returncode = 0
    (tmp_path / "potential.yace").touch()

    trainer.train(structures)

    mock_dm_instance.select_active_set.assert_called_once()
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
    mock_dm_instance.create_dataset.return_value = tmp_path / "dataset.pckl.gzip"

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
    mock_dm_instance.create_dataset.return_value = tmp_path / "dataset.pckl.gzip"

    mock_run.return_value.returncode = 0
    # Ensure no .yace files exist
    for f in tmp_path.glob("*.yace"):
        f.unlink()

    with pytest.raises(FileNotFoundError, match="did not produce a .yace file"):
        trainer.train(sample_structures)

def test_select_active_set_fallback(trainer_config: TrainerConfig, sample_structures: list[Structure], tmp_path: Path) -> None:
    trainer = PacemakerTrainer(work_dir=tmp_path, config=trainer_config)

    # We rely on real DatasetManager creating a temp file (which fails if no ase/pace_collect)
    # But here we just want to test fallback logic in PacemakerTrainer.select_active_set
    # It calls self.dataset_manager.create_dataset
    # We should mock dataset_manager

    trainer.dataset_manager = MagicMock()

    # structures len=1
    # requesting count=1

    selected = list(trainer.select_active_set(sample_structures, count=1))

    assert len(selected) == 1
    assert selected[0] == sample_structures[0]
    trainer.dataset_manager.create_dataset.assert_called_once()
