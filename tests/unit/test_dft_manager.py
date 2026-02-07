from unittest.mock import patch

import numpy as np
import pytest
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

def test_dft_manager_init_invalid_command() -> None:
    params = {
        "command": "pw.x; rm -rf /", # Dangerous command
        "pseudo_dir": "/tmp", # noqa: S108
        "pseudopotentials": {"Si": "Si.upf"}
    }
    with pytest.raises(ValueError, match="Invalid characters in command"):
        DFTManager(params)

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

def test_dft_manager_kpoint_grid_logic() -> None:
    # Test k-point grid logic for different cells
    params = {
        "command": "pw.x",
        "pseudo_dir": "/tmp",
        "pseudopotentials": {"Si": "Si.upf"},
        "kspacing": 0.2
    }
    oracle = DFTManager(params)

    # 1. Cubic cell 10x10x10
    # b = 2pi/10 = 0.6283. N = ceil(0.6283/0.2) = 4
    structure_cubic = Structure(
        positions=np.zeros((1, 3)),
        cell=np.eye(3) * 10.0,
        species=["Si"]
    )
    with patch("mlip_autopipec.infrastructure.oracle.dft_manager.Espresso") as MockEspresso:
        MockEspresso.return_value.get_potential_energy.return_value = -1.0
        MockEspresso.return_value.get_forces.return_value = np.zeros((1, 3))
        MockEspresso.return_value.get_stress.return_value = np.zeros(6)

        oracle.compute(structure_cubic)
        _, kwargs = MockEspresso.call_args
        assert kwargs["kpts"] == (4, 4, 4)

    # 2. Orthorhombic cell 5x10x20
    # b1 = 2pi/5 = 1.2566 -> 7
    # b2 = 2pi/10 = 0.6283 -> 4
    # b3 = 2pi/20 = 0.3141 -> 2
    structure_ortho = Structure(
        positions=np.zeros((1, 3)),
        cell=np.diag([5.0, 10.0, 20.0]),
        species=["Si"]
    )
    with patch("mlip_autopipec.infrastructure.oracle.dft_manager.Espresso") as MockEspresso:
        MockEspresso.return_value.get_potential_energy.return_value = -1.0
        MockEspresso.return_value.get_forces.return_value = np.zeros((1, 3))
        MockEspresso.return_value.get_stress.return_value = np.zeros(6)

        oracle.compute(structure_ortho)
        _, kwargs = MockEspresso.call_args
        assert kwargs["kpts"] == (7, 4, 2)

def test_dft_manager_persistent_failure() -> None:
    # Test behavior when Espresso raises persistent error
    params = {
        "command": "pw.x",
        "pseudo_dir": "/tmp",
        "pseudopotentials": {"Si": "Si.upf"}
    }
    oracle = DFTManager(params)
    structure = Structure(
        positions=np.zeros((1, 3)),
        cell=np.eye(3) * 10.0,
        species=["Si"]
    )

    with patch("mlip_autopipec.infrastructure.oracle.dft_manager.Espresso") as MockEspresso:
        # Always fail
        MockEspresso.return_value.get_potential_energy.side_effect = CalculationFailed("Fatal error")

        with pytest.raises(RuntimeError, match="DFT calculation failed after 2 retries"):
            oracle.compute(structure)

def test_dft_manager_invalid_structure_input() -> None:
    # Test that unknown species raises error (if ASE validation kicks in or we mock it)
    params = {
        "command": "pw.x",
        "pseudo_dir": "/tmp",
        "pseudopotentials": {"Si": "Si.upf"}
    }
    oracle = DFTManager(params)
    # Structure with species "X" which is valid string but maybe no pseudopotential
    structure = Structure(
        positions=np.zeros((1, 3)),
        cell=np.eye(3) * 10.0,
        species=["X"]
    )

    # In compute, Atoms(...) is created. If symbol is invalid, it might raise.
    # But ASE Atoms usually accepts "X".
    # However, if pseudopotentials are missing "X", Espresso init might raise or run might fail.
    # Espresso init checks pseudos? ASE Espresso does not check keys against structure immediately unless we run it.

    # We can simulate ASE raising error during run due to missing pseudo.

    with patch("mlip_autopipec.infrastructure.oracle.dft_manager.Espresso") as MockEspresso:
        MockEspresso.side_effect = RuntimeError("Missing pseudopotential for X")

        with pytest.raises(RuntimeError, match="Missing pseudopotential for X"):
            oracle.compute(structure)
