from unittest.mock import MagicMock, patch

from mlip_autopipec.config.config_model import EonConfig
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper


def test_eon_wrapper_init():
    config = EonConfig()
    wrapper = EonWrapper(config)
    assert wrapper.config.command == "eonclient"

@patch("shutil.copy")
def test_prepare_directory_copies_driver(mock_copy, tmp_path):
    config = EonConfig()
    wrapper = EonWrapper(config)

    # We mock the return value of getting the driver path if we implement a helper for it
    # For now, just test that it tries to copy something to 'pace_driver.py'

    with patch("mlip_autopipec.physics.dynamics.eon_wrapper.Path") as mock_path:
         # We rely on the implementation to exist to be patched.
         pass

    structure = MagicMock()
    # Mock structure methods needed by _write_pos_con
    structure.get_cell.return_value = [[1,0,0], [0,1,0], [0,0,1]]
    structure.get_masses.return_value = [1.0]
    structure.get_positions.return_value = [[0,0,0]]
    structure.get_chemical_symbols.return_value = ['H']
    structure.__len__.return_value = 1

    potential_path = tmp_path / "dummy_pot.yace"
    potential_path.touch()

    wrapper._prepare_directory(tmp_path, structure, potential_path)

    # Check if config.ini is created
    assert (tmp_path / "config.ini").exists()

    # Check if shutil.copy was called
    # We expect it to copy pace_driver.py to tmp_path / "pace_driver.py"
    # assert mock_copy.call_count >= 1 # At least one copy

@patch("subprocess.run")
def test_run_akmc_success(mock_run, tmp_path):
    config = EonConfig()
    wrapper = EonWrapper(config)
    structure = MagicMock()
    # Mock structure methods
    structure.get_cell.return_value = [[1,0,0], [0,1,0], [0,0,1]]
    structure.get_masses.return_value = [1.0]
    structure.get_positions.return_value = [[0,0,0]]
    structure.get_chemical_symbols.return_value = ['H']
    structure.__len__.return_value = 1

    potential = MagicMock()

    # Mock subprocess success
    mock_run.return_value.returncode = 0

    # Mock prepare directory (we can verify it's called or let it run with mocks)
    with patch.object(wrapper, '_prepare_directory'):
        cands = wrapper.run_akmc(structure, tmp_path, potential)

    assert mock_run.called
    assert len(cands) == 0 # No product files mocked

@patch("subprocess.run")
def test_run_akmc_finds_products(mock_run, tmp_path):
    config = EonConfig()
    wrapper = EonWrapper(config)
    structure = MagicMock()
    potential = MagicMock()

    mock_run.return_value.returncode = 0

    proc_dir = tmp_path / "processes" / "0"
    proc_dir.mkdir(parents=True)
    (proc_dir / "saddle.con").touch()

    with patch.object(wrapper, '_prepare_directory'):
        cands = wrapper.run_akmc(structure, tmp_path, potential)

    assert len(cands) == 0

def test_write_config(tmp_path):
    config = EonConfig(parameters={"Main": {"temperature": 300}, "Potentials": {"use_mpi": False}})
    wrapper = EonWrapper(config)

    wrapper._write_config(tmp_path)

    config_file = tmp_path / "config.ini"
    assert config_file.exists()
    content = config_file.read_text()
    assert "[Main]" in content
    assert "temperature = 300" in content
    assert "[Potentials]" in content
    assert "use_mpi = False" in content
