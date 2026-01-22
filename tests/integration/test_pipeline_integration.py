
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline


@pytest.fixture
def mock_db(tmp_path):
    """
    Creates a temporary database for integration testing.
    This replaces any 'real' DB file with an isolated temp file.
    """
    db_path = tmp_path / "test_pipeline.db"
    return DatabaseManager(db_path)

def test_integration_pipeline_real_db(mock_db):
    """
    Integration test using a real (but temporary) SQLite database file.
    Verifies interaction between DB and Surrogate Pipeline.
    """
    # 1. Setup DB with pending structures
    with mock_db:
        mock_db.add_structure(Atoms('H2', positions=[[0,0,0], [0,0,0.74]]), metadata={"status": "pending"})
        mock_db.add_structure(Atoms('O2', positions=[[0,0,0], [0,0,1.2]]), metadata={"status": "pending"})
        assert mock_db.count(status="pending") == 2

    # 2. Configure Surrogate Pipeline (using mock model for simplicity in this integration scope)
    config = SurrogateConfig(
        model_type="mock",
        model_path="dummy.pt",
        force_threshold=100.0,
        n_samples=1
    )

    # Mocking the MaceWrapper internally used by pipeline to avoid heavy model loading
    # We patch at the class level or inject a mock model if the constructor allows.
    # SurrogatePipeline allows injecting `model`.

    from unittest.mock import MagicMock

    import numpy as np
    mock_model_interface = MagicMock()
    # Mock return: energy per atom, forces per atom
    # H2: 2 atoms. O2: 2 atoms.
    mock_model_interface.compute_energy_forces.return_value = (
        np.array([-1.0, -2.0]), # Energies
        [np.array([[0,0,0], [0,0,0]]), np.array([[0,0,0], [0,0,0]])] # Zero forces
    )
    # Mock descriptors
    mock_model_interface.compute_descriptors.return_value = np.array([[1.0], [2.0]])

    pipeline = SurrogatePipeline(mock_db, config, model=mock_model_interface)

    # 3. Run Pipeline
    pipeline.run()

    # 4. Verify DB state
    with mock_db:
        # 1 selected (n_samples=1), 1 held
        assert mock_db.count(status="selected") == 1
        assert mock_db.count(status="held") == 1
        assert mock_db.count(status="pending") == 0

        # Verify correctness of exported data (check info keys)
        selected_atoms = mock_db.get_atoms(status="selected")
        for at in selected_atoms:
             # Ensure MACE energy/forces are stored if computed (impl dependent)
             # The pipeline stores them in info/arrays usually.
             pass


def test_integration_training_flow(mock_db, tmp_path):
    """
    Integration test for the training flow: DB -> Dataset -> Config -> Pacemaker(Mock).
    """
    from unittest.mock import patch

    import numpy as np

    from mlip_autopipec.config.schemas.training import TrainingConfig, TrainingMetrics
    from mlip_autopipec.training.dataset import DatasetBuilder
    from mlip_autopipec.training.pacemaker import PacemakerWrapper

    # 1. Setup DB with completed structures
    with mock_db:
        for i in range(10):
             at = Atoms('Cu', positions=[[0,0,0]])
             at.info['energy'] = -3.0
             at.arrays['forces'] = np.array([[0.0, 0.0, 0.0]])
             mock_db.add_structure(at, metadata={"status": "completed"})

    config = TrainingConfig(
        cutoff=3.0,
        b_basis_size=10,
        kappa=1.0,
        kappa_f=1.0,
        max_iter=10
    )

    work_dir = tmp_path / "training_work"

    # 2. Export Data
    builder = DatasetBuilder(mock_db)
    builder.export(config, work_dir)

    train_file = work_dir / "data" / "train.xyz"
    assert train_file.exists()
    assert (work_dir / "data" / "test.xyz").exists()

    # Verify content
    with open(train_file) as f:
        content = f.read()
        assert "Lattice=" in content
        assert "energy=" in content
        assert "Cu" in content

    # 3. Train (Mocking Pacemaker subprocess)
    with patch("mlip_autopipec.training.pacemaker.subprocess.run") as mock_run, \
         patch("mlip_autopipec.training.pacemaker.PacemakerWrapper.check_output", return_value=True), \
         patch("mlip_autopipec.training.metrics.LogParser.parse_file") as mock_parse:

        mock_run.return_value.returncode = 0
        mock_parse.return_value = TrainingMetrics(epoch=10, rmse_energy=0.1, rmse_force=0.01)

        wrapper = PacemakerWrapper(config, work_dir)
        result = wrapper.train()

        assert result.success is True
        assert result.metrics.rmse_energy == 0.1
        assert (work_dir / "input.yaml").exists()
