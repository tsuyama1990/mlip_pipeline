import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, SCFError
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.dft.input_gen import InputGenerator

# UAT Scenarios

@pytest.fixture
def uat_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0.0, 0.0, 0.0], [1.3, 1.3, 1.3]]),
        cell=np.array([[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]),
        pbc=(True, True, True)
    )

@pytest.fixture
def uat_config(tmp_path):
    p = tmp_path / "Si.UPF"
    p.touch()
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": p},
        ecutwfc=30.0,
        kspacing=0.2
    )

@patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run")
@patch("mlip_autopipec.physics.dft.qe_runner.Parser")
def test_uat_scenario_3_1_standard_calc(MockParser, mock_run, uat_structure, uat_config, tmp_path):
    """Scenario 3.1: Standard DFT Calculation"""
    # Simulate success
    mock_run.return_value.stdout = "JOB DONE"
    expected_result = DFTResult(
        energy=-200.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros((3, 3))
    )
    MockParser.parse.return_value = expected_result

    runner = QERunner(working_dir=tmp_path)
    result = runner.run(uat_structure, uat_config)

    assert result == expected_result
    assert result.energy == -200.0
    # Forces should be near zero (checked by mock return)

@patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run")
@patch("mlip_autopipec.physics.dft.qe_runner.Parser")
def test_uat_scenario_3_2_self_healing(MockParser, mock_run, uat_structure, uat_config, tmp_path):
    """Scenario 3.2: Self-Healing"""
    # Simulate failure then success

    # 1st call raises SCFError
    # 2nd call returns result

    final_result = DFTResult(
        energy=-200.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros((3, 3))
    )

    MockParser.parse.side_effect = [SCFError("Convergence not achieved"), final_result]

    # We want to verify that config was modified.
    # We can inspect the arguments passed to subprocess.run (or InputGenerator)
    # But QERunner instantiates InputGenerator internally or uses static method.
    # Let's patch InputGenerator too to spy on it.

    with patch("mlip_autopipec.physics.dft.qe_runner.InputGenerator") as MockInputGen:
        MockInputGen.generate.return_value = "dummy content"
        runner = QERunner(working_dir=tmp_path)
        # Force unstable param
        uat_config.mixing_beta = 0.9

        result = runner.run(uat_structure, uat_config)

        assert result == final_result
        assert mock_run.call_count == 2

        # Check that InputGenerator was called with different configs
        assert MockInputGen.generate.call_count == 2
        call_args_list = MockInputGen.generate.call_args_list

        # First call: beta 0.9
        config1 = call_args_list[0][0][1] # (structure, config)
        assert config1.mixing_beta == 0.9

        # Second call: beta should be reduced (e.g. 0.7*0.9 or fixed value 0.3)
        # Depending on RecoveryHandler logic. SPEC says "mixing_beta": 0.3
        config2 = call_args_list[1][0][1]
        assert config2.mixing_beta < 0.9

def test_uat_scenario_3_3_automatic_kpoints(uat_structure, uat_config):
    """Scenario 3.3: Automatic K-Point Grid"""
    # This relies on InputGenerator logic

    # Case A: Small cell (5.43) -> High K
    # kspacing=0.2. N = 2*pi/(5.43*0.2) ~ 5.7 -> 6
    content_a = InputGenerator.generate(uat_structure, uat_config)
    assert "6 6 6" in content_a

    # Case B: Large supercell
    large_struct = uat_structure.model_copy()
    large_struct.cell = np.eye(3) * 20.0
    large_struct.positions = np.zeros((2, 3)) # Dummy

    # N = 2*pi/(20*0.2) ~ 1.57 -> 2
    content_b = InputGenerator.generate(large_struct, uat_config)
    assert "2 2 2" in content_b
