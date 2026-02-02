import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.config_model import TrainingConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer


@pytest.fixture
def trainer_config(tmp_path: Path) -> TrainingConfig:
    data_path = tmp_path / "data.pckl"
    data_path.touch()
    return TrainingConfig(
        dataset_path=data_path,
        max_epochs=5,
        command="mock_pace"
    )

def test_trainer_mock_mode(trainer_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")
    monkeypatch.chdir(tmp_path)

    trainer = PacemakerTrainer(trainer_config)
    output = trainer.train(trainer_config.dataset_path)

    assert output.exists()
    assert output.name == "output_potential.yace"

@patch("subprocess.run")
def test_trainer_command_construction(mock_run: MagicMock, trainer_config: TrainingConfig, tmp_path: Path) -> None:
    trainer = PacemakerTrainer(trainer_config)

    # We must ensure output file exists or mock logic to avoid FileNotFoundError
    # Actually, if we mock subprocess, we must also mock existence check
    # or ensure we don't hit it if check=True raises?
    # No, after subprocess, it checks file existence.
    # So we should touch the file in the test before calling train, or mock Path.exists

    # Let's just verify call args and ignore the FileNotFoundError at the end
    with contextlib.suppress(FileNotFoundError):
        trainer.train(trainer_config.dataset_path)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "mock_pace"
    assert "--dataset" in args
    assert "--max-epochs" in args
    assert "5" in args
