from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.components.oracle.qe import QECalculator, QEOracle
from mlip_autopipec.domain_models.config import QEOracleConfig
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def qe_config() -> QEOracleConfig:
    return QEOracleConfig(
        name=OracleType.QE, kspacing=0.1, mixing_beta=0.7, ecutwfc=30.0, ecutrho=150.0
    )


@pytest.fixture
def structure() -> Structure:
    # Create a simple Si structure
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    return Structure.from_ase(atoms)


def test_qe_initialization(qe_config: QEOracleConfig) -> None:
    oracle = QEOracle(qe_config)
    assert oracle.name == OracleType.QE
    assert oracle.config.kspacing == 0.1


@patch("mlip_autopipec.components.oracle.qe.Espresso")
def test_qe_compute_success(
    mock_espresso_cls: MagicMock, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Setup mock calculator
    mock_calc = MagicMock(spec=Calculator)
    mock_calc.get_potential_energy.return_value = -100.0
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.get_stress.return_value = np.zeros(6)  # Voigt
    mock_calc.parameters = {}

    mock_espresso_cls.return_value = mock_calc

    oracle = QEOracle(qe_config)
    results = list(oracle.compute([structure]))

    assert len(results) == 1
    s = results[0]
    assert s.energy == -100.0
    assert s.forces is not None
    assert np.allclose(s.forces, np.zeros((2, 3)))
    assert s.stress is not None
    assert np.allclose(s.stress, np.zeros((3, 3)))
    assert "qe_params" in s.tags  # Provenance check if we add it


@patch("mlip_autopipec.components.oracle.qe.Espresso")
def test_qe_compute_healing_success(
    mock_espresso_cls: MagicMock, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Setup mock calculator to fail once, then succeed
    mock_calc = MagicMock(spec=Calculator)
    mock_calc.parameters = {"mixing_beta": 0.7}

    # Side effect for get_potential_energy: raise Error first, then return value
    # We use chain/repeat because Structure.from_ase will call get_potential_energy again
    from itertools import chain, repeat

    mock_calc.get_potential_energy.side_effect = chain(
        [Exception("Convergence error")], repeat(-100.0)
    )
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.get_stress.return_value = np.zeros(6)

    mock_espresso_cls.return_value = mock_calc

    oracle = QEOracle(qe_config)
    results = list(oracle.compute([structure]))

    assert len(results) == 1
    s = results[0]
    assert s.energy == -100.0
    # Verify healing happened (mixing_beta reduced)
    # The heuristic in Healer reduces mixing_beta to 0.3
    assert mock_calc.parameters["mixing_beta"] == 0.3


@patch("mlip_autopipec.components.oracle.qe.Espresso")
def test_qe_compute_healing_failure(
    mock_espresso_cls: MagicMock, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Setup mock calculator to always fail
    mock_calc = MagicMock(spec=Calculator)
    mock_calc.parameters = {"mixing_beta": 0.7}
    mock_calc.get_potential_energy.side_effect = Exception("Persistent error")

    mock_espresso_cls.return_value = mock_calc

    oracle = QEOracle(qe_config)
    results = list(oracle.compute([structure]))

    # Should be empty because structure is skipped
    assert len(results) == 0


def test_qe_calculator_setup(qe_config: QEOracleConfig) -> None:
    """Test QECalculator internal logic if exposed, or verify input generation via mock."""
    # Since QECalculator wraps Espresso, we can test that it passes correct params.
    with patch("mlip_autopipec.components.oracle.qe.Espresso") as mock_espresso_cls:
        qe_calc = QECalculator(qe_config)
        # Create a dummy atoms
        atoms = Atoms("H")
        qe_calc.calculate(atoms)

        # Verify Espresso initialized with correct params
        mock_espresso_cls.assert_called()
        call_kwargs = mock_espresso_cls.call_args[1]
        assert call_kwargs["ecutwfc"] == 30.0
        assert call_kwargs["kspacing"] == 0.1
        assert call_kwargs["tprnfor"] is True
        assert call_kwargs["tstress"] is True
