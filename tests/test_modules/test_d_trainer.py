from pathlib import Path
from unittest.mock import MagicMock

from ase.db import connect
from ase.io import read

from mlip_autopipec.modules.d_trainer import PacemakerTrainer
from mlip_autopipec.schemas.user_config import TrainerConfig
from mlip_autopipec.utils.pacemaker_utils import generate_pacemaker_input


def test_generate_pacemaker_input() -> None:
    """Test the generation of the pacemaker.in file content."""
    config = TrainerConfig(
        radial_basis="something",
        max_body_order=4,
        loss_weights={"energy": 1.0, "forces": 100.0, "stress": 0.0},
    )

    input_content = generate_pacemaker_input(config, "structures.extxyz")

    assert 'radial_basis = "something"' in input_content
    assert "max_body_order = 4" in input_content
    assert "l_max = 2" in input_content
    assert "loss_weights = {'energy': 1.0, 'forces': 100.0, 'stress': 0.0}" in input_content
    assert 'dataset_path = "structures.extxyz"' in input_content
    assert "delta_learning = True" in input_content


def test_train_potential(mocker: MagicMock, tmp_path: Path) -> None:
    """Test the train_potential method."""
    # Create a dummy database
    db_path = tmp_path / "test.db"
    with connect(db_path) as db:  # type: ignore[no-untyped-call]
        db.write(read("tests/test_files/si_bulk.xyz"))

    # Mock subprocess.run
    mock_subprocess = mocker.patch("subprocess.run")

    config = TrainerConfig(
        radial_basis="something",
        max_body_order=4,
        loss_weights={"energy": 1.0, "forces": 100.0, "stress": 0.0},
    )
    trainer = PacemakerTrainer(config)
    trainer.train_potential(str(db_path), tmp_path)

    mock_subprocess.assert_called_once()
