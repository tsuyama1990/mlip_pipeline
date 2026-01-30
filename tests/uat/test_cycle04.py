from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from ase.build import bulk

from mlip_autopipec.domain_models.config import Config, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.training import TrainingConfig

@pytest.fixture
def mock_structures():
    s1 = Structure.from_ase(bulk("Ti"))
    s1.properties = {"energy": -100.0, "forces": [[0,0,0]]*len(s1.symbols), "stress": [0]*6}
    return [s1]

def test_uat_c04_01_full_training_loop(tmp_path):
    """
    UAT-C04-01: Full Training Loop
    Verifies that the Orchestrator (or a manual script using components)
    can go from Structures -> Potential.
    """
    from mlip_autopipec.physics.training.dataset import DatasetManager
    from mlip_autopipec.physics.training.pacemaker import PacemakerRunner

    # 1. Setup
    dataset_mgr = DatasetManager(work_dir=tmp_path)
    runner = PacemakerRunner(work_dir=tmp_path)

    structures = [Structure.from_ase(bulk("Ti"))]
    structures[0].properties = {"energy": -100.0, "forces": [[0,0,0]]*len(structures[0].symbols), "stress": [0]*6}

    train_config = TrainingConfig(batch_size=10, max_epochs=1)
    pot_config = PotentialConfig(elements=["Ti"], cutoff=4.0)

    # 2. Execution (Mocking external tools)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        def side_effect(*args, **kwargs):
             # Simulate pace_train outputting to stdout (which is redirected to file)
             if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
                 kwargs["stdout"].write("RMSE Energy: 0.01\nRMSE Force: 0.02\n")
             # Also create the potential file which pace_train would do
             (tmp_path / "output_potential.yace").touch()

        mock_run.side_effect = side_effect

        # A) Dataset Conversion
        dataset_path = dataset_mgr.convert(structures, tmp_path / "train.pckl.gzip")

        # B) Training
        result = runner.train(dataset_path, train_config, pot_config)

        # 3. Validation
        assert result.status == JobStatus.COMPLETED
        assert result.potential.path.exists()
        assert result.validation_metrics["energy_rmse"] == 0.01
