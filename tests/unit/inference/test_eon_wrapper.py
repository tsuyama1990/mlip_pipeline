import pytest
from unittest.mock import patch
from pathlib import Path
from mlip_autopipec.config.schemas.inference import EONConfig
from ase import Atoms

@pytest.fixture
def eon_config():
    return EONConfig(
        eon_executable=Path("/usr/bin/eonclient"),
        job="process_search",
        temperature=400.0,
        pot_name="pace_driver",
        parameters={"quiet": True}
    )

def test_eon_wrapper_init(eon_config, tmp_path):
    try:
        from mlip_autopipec.inference.eon import EONWrapper
    except ImportError:
        pytest.fail("EONWrapper not implemented")

    wrapper = EONWrapper(eon_config, tmp_path)
    assert wrapper.config == eon_config
    assert wrapper.work_dir == tmp_path

def test_eon_wrapper_write_config(eon_config, tmp_path):
    try:
        from mlip_autopipec.inference.eon import EONWrapper
    except ImportError:
        pytest.fail("EONWrapper not implemented")

    wrapper = EONWrapper(eon_config, tmp_path)
    wrapper._write_config()

    config_file = tmp_path / "config.ini"
    assert config_file.exists()
    content = config_file.read_text()

    # Check key parameters
    assert "job = process_search" in content
    assert "temperature = 400.0" in content
    assert "potential = pace_driver" in content

@patch("subprocess.run")
def test_eon_wrapper_run_success(mock_run, eon_config, tmp_path):
    try:
        from mlip_autopipec.inference.eon import EONWrapper
    except ImportError:
        pytest.fail("EONWrapper not implemented")

    mock_run.return_value.returncode = 0

    wrapper = EONWrapper(eon_config, tmp_path)
    atoms = Atoms("H2", positions=[[0,0,0], [1,0,0]])
    potential_path = Path("pot.yace")

    with patch.object(wrapper, "_write_pos_con"):
        result = wrapper.run(atoms, potential_path)

    assert result.succeeded is True
    assert result.max_gamma_observed < eon_config.parameters.get("uncertainty_threshold", 5.0)

@patch("subprocess.run")
def test_eon_wrapper_run_halt(mock_run, eon_config, tmp_path):
    try:
        from mlip_autopipec.inference.eon import EONWrapper
    except ImportError:
        pytest.fail("EONWrapper not implemented")

    # Simulate Halt code 100
    mock_run.return_value.returncode = 100

    wrapper = EONWrapper(eon_config, tmp_path)
    atoms = Atoms("H2")
    potential_path = Path("pot.yace")

    # Mock existence of bad_structure.con
    bad_structure_file = tmp_path / "bad_structure.con"
    bad_structure_file.touch()

    with patch.object(wrapper, "_write_pos_con"):
        result = wrapper.run(atoms, potential_path)

    assert result.succeeded is True
    assert len(result.uncertain_structures) == 1
    assert result.uncertain_structures[0] == bad_structure_file
