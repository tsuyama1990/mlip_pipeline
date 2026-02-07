import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ase.calculators.calculator import CalculationFailed

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.oracle import DFTManager


def test_dft_manager_init() -> None:
    params = {
        "command": "pw.x",
        "pseudo_dir": "/tmp", # noqa: S108
        "pseudopotentials": {"Si": "Si.upf"},
        "kspacing": 0.05
    }
    oracle = DFTManager(params)
    assert oracle.command == "pw.x"
    assert oracle.kspacing == 0.05
    assert oracle.smearing_width == 0.02 # default

def test_dft_manager_compute_logic() -> None:
    # Test logic with mocked ASE calculator
    params = {
        "command": "pw.x",
        "pseudo_dir": "/tmp", # noqa: S108
        "pseudopotentials": {"Si": "Si.upf"}
    }
    oracle = DFTManager(params)

    structure = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3) * 5.0,
        species=["Si"]
    )

    with patch("mlip_autopipec.infrastructure.oracle.dft_manager.Espresso") as MockEspresso:
        # Configure mock
        mock_calc = MockEspresso.return_value
        # ase.Atoms.get_potential_energy calls calc.get_potential_energy
        mock_calc.get_potential_energy.return_value = -10.0
        mock_calc.get_forces.return_value = np.zeros((1, 3))
        mock_calc.get_stress.return_value = np.zeros(6) # Voigt form usually

        result_struct = oracle.compute(structure)

        assert result_struct.energy == -10.0
        assert result_struct.forces is not None
        assert np.allclose(result_struct.forces, 0.0)

        # Verify Espresso called with correct params
        MockEspresso.assert_called_once()
        _, kwargs = MockEspresso.call_args

        # We check kwargs passed to Espresso init
        assert "profile" in kwargs
        assert kwargs["pseudopotentials"] == {"Si": "Si.upf"}
        assert "kpts" in kwargs
        # kpts depends on kspacing=0.04 (default)
        # b = 2pi/5 = 1.2566. N = ceil(1.2566/0.04) = 32
        assert kwargs["kpts"] == (32, 32, 32)

def test_dft_manager_retry_logic() -> None:
    # Test self-healing logic
    params = {
        "command": "pw.x",
        "pseudo_dir": "/tmp", # noqa: S108
        "pseudopotentials": {"Si": "Si.upf"}
    }
    oracle = DFTManager(params)
    structure = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3) * 5.0,
        species=["Si"]
    )

    with patch("mlip_autopipec.infrastructure.oracle.dft_manager.Espresso") as MockEspresso:
        mock_calc = MockEspresso.return_value

        # First call raises CalculationFailed
        # Second call succeeds
        mock_calc.get_potential_energy.side_effect = [CalculationFailed("SCF not converged"), -10.0]
        mock_calc.get_forces.return_value = np.zeros((1, 3))
        mock_calc.get_stress.return_value = np.zeros(6)

        result_struct = oracle.compute(structure)

        assert result_struct.energy == -10.0

        # Verify Espresso was called twice
        assert MockEspresso.call_count == 2

        # Check that the second call had modified parameters
        call_args_list = MockEspresso.call_args_list

        # First call: default mixing_beta = 0.7
        _, kwargs1 = call_args_list[0]
        assert kwargs1["input_data"]["electrons"]["mixing_beta"] == 0.7

        # Second call: reduced mixing_beta = 0.35
        _, kwargs2 = call_args_list[1]
        assert kwargs2["input_data"]["electrons"]["mixing_beta"] == 0.35
