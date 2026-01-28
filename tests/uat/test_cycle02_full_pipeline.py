from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.schemas.training import TrainingMetrics, TrainingResult


def test_run_cycle_02_full_pipeline(tmp_path):
    runner = CliRunner()

    config_content = """
target_system:
  name: Al
  elements: [Al]
  composition: {Al: 1.0}
  crystal_structure: fcc

generator_config:
  number_of_structures: 2
  seed: 42
  distortion:
    enabled: true
    n_strain_steps: 1
    n_rattle_steps: 1

dft:
  command: "echo 'mock'"
  pseudopotential_dir: "."

training_config:
  cutoff: 4.0
  b_basis_size: 10
  kappa: 0.5
  kappa_f: 1.0
  batch_size: 10
  max_iter: 10

runtime:
  work_dir: "_work_test"
  database_path: "test.db"
"""

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("input.yaml").write_text(config_content)
        # Create dummy UPF file to satisfy validation
        Path("dummy.UPF").touch()

        with patch("mlip_autopipec.training.pacemaker.PacemakerWrapper.train") as mock_train:
            # Mock successful training
            mock_train.return_value = TrainingResult(
                success=True,
                potential_path=str(Path("_work_test") / "training" / "output.yace"),
                metrics=TrainingMetrics(epoch=10, rmse_energy=0.01, rmse_force=0.1)
            )

            result = runner.invoke(app, ["run", "cycle-02", "--config", "input.yaml", "--mock-dft"])

            assert result.exit_code == 0
            assert "Step 1: Structure Generation" in result.stdout
            assert "Generated 2 structures" in result.stdout
            assert "Step 2: Oracle Calculation" in result.stdout
            assert "Running in MOCK DFT mode" in result.stdout
            assert "Step 3: Training" in result.stdout
            assert "Training Successful" in result.stdout

            mock_train.assert_called_once()
