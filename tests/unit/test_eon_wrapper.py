import configparser
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.config_model import Config, EonConfig, TrainingConfig
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper


@pytest.fixture
def mock_config():
    conf = MagicMock(spec=Config)
    conf.eon = EonConfig(
        command="eonclient",
        parameters={"temperature": 300, "random_seed": 123}
    )
    conf.training = MagicMock(spec=TrainingConfig)
    conf.training.dataset_path = Path("data.xyz")
    return conf

def test_eon_wrapper_setup(mock_config, tmp_path):
    wrapper = EonWrapper(config=mock_config)

    # Create dummy potential file
    potential_path = tmp_path / "test.yace"
    potential_path.write_text("dummy potential")

    structure = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10])

    work_dir = tmp_path / "eon_run"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        wrapper.run_akmc(potential_path, structure, work_dir)

        # Check config.ini creation
        config_ini = work_dir / "config.ini"
        assert config_ini.exists()

        parser = configparser.ConfigParser()
        parser.read(config_ini)
        found = False
        for section in parser.sections():
            if "temperature" in parser[section]:
                assert parser[section]["temperature"] == "300"
                found = True
        assert found

        # Check files
        assert (work_dir / "pos.con").exists()
        assert (work_dir / "potential.yace").exists()
        assert (work_dir / "pace_driver.py").exists()

        # Check command execution
        mock_run.assert_called()
        args = mock_run.call_args[0][0]
        assert "eonclient" in args

def test_eon_wrapper_halt_detection(mock_config, tmp_path):
    wrapper = EonWrapper(config=mock_config)

    potential_path = tmp_path / "test.yace"
    potential_path.write_text("dummy")

    structure = Atoms("Cu", positions=[[0, 0, 0]])
    work_dir = tmp_path / "eon_run_halt"

    with patch("subprocess.run") as mock_run:
        # Simulate exit code 100
        mock_run.return_value.returncode = 100

        exit_code = wrapper.run_akmc(potential_path, structure, work_dir)
        assert exit_code == 100
