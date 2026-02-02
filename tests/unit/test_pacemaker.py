from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config import TrainingConfig
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer


def test_pacemaker_train_command(temp_dir: Path) -> None:
    dataset = temp_dir / "data.pckl"
    dataset.touch()
    config = TrainingConfig(dataset_path=dataset, max_epochs=10, command="pace_train")
    trainer = PacemakerTrainer(config)
    output_dir = temp_dir / "output"

    with patch("subprocess.run") as mock_run:
        # Mock successful run
        mock_run.return_value.returncode = 0

        # We also need to mock output file existence check
        with patch("pathlib.Path.exists", return_value=True):
            trainer.train(dataset, previous_potential=None, output_dir=output_dir)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pace_train"
        assert "--dataset" in cmd
        assert "--max_num_epochs" in cmd
        assert "10" in cmd
        assert "--output_dir" in cmd


def test_pacemaker_mock_mode(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYACEMAKER_MOCK_MODE", "1")
    dataset = temp_dir / "data.pckl"
    dataset.touch()
    config = TrainingConfig(dataset_path=dataset)
    trainer = PacemakerTrainer(config)
    output_dir = temp_dir / "output"

    with patch("subprocess.run") as mock_run:
        output = trainer.train(dataset, previous_potential=None, output_dir=output_dir)
        mock_run.assert_not_called()
        assert output.name == "output_potential.yace"
        assert output.exists()
