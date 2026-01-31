from unittest.mock import MagicMock, patch
from mlip_autopipec.cli import commands

def test_uat_full_training_loop(tmp_path):
    """
    UAT-C04-01: Full Training Loop
    """
    config_path = tmp_path / "config.yaml"
    # Create a minimal valid config
    config_content = """
project_name: "TestProject"
potential:
  elements: ["Ti", "O"]
  cutoff: 5.0
md:
  temperature: 300
  n_steps: 100
  ensemble: "NVT"
structure_gen:
  strategy: "bulk"
  element: "Ti"
  crystal_structure: "hcp"
  lattice_constant: 2.95
training:
  max_epochs: 10
    """
    config_path.write_text(config_content)

    structures_path = tmp_path / "structures.xyz"
    structures_path.touch()

    # Mocking dependencies
    with patch("mlip_autopipec.cli.commands.DatasetManager") as MockDM, \
         patch("mlip_autopipec.cli.commands.PacemakerRunner") as MockPR, \
         patch("mlip_autopipec.cli.commands.io.load_structures") as MockLoad:

        # Setup Mocks
        mock_dm_instance = MockDM.return_value
        mock_dm_instance.create_dataset.return_value = tmp_path / "train.pckl.gzip"

        mock_pr_instance = MockPR.return_value
        mock_result = MagicMock()
        mock_result.potential_path = tmp_path / "potential.yace"
        mock_result.validation_metrics = {"rmse_energy": 0.005}
        mock_result.status.value = "COMPLETED"
        # Need to make sure status equality check works. MagicMock might not equal Enum.
        # But if we access .value in code? No, we access result.status directly in commands.py
        # "if result.status == JobStatus.COMPLETED:"

        from mlip_autopipec.domain_models.job import JobStatus
        mock_result.status = JobStatus.COMPLETED

        mock_pr_instance.train.return_value = mock_result

        MockLoad.return_value = [MagicMock()]

        # Execute command
        commands.train_model(config_path, structures_path)

        # Verifications
        mock_dm_instance.create_dataset.assert_called_once()
        mock_pr_instance.train.assert_called_once()
