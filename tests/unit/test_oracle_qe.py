from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.components.oracle.qe import QECalculator, QEOracle
from mlip_autopipec.domain_models.config import (
    HEALER_MIXING_BETA_TARGET,
    QEOracleConfig,
)
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.domain_models.structure import Structure


# Define a Fake Calculator that behaves like Espresso but runs in memory
class FakeEspresso(Calculator):
    def __init__(self, failure_mode: str = "parameter_sensitive", **kwargs: Any) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.parameters = kwargs
        # Ensure failure_mode is in parameters so it survives Healing reconstruction
        self.parameters["failure_mode"] = failure_mode
        self.failure_mode = failure_mode
        self.implemented_properties = ["energy", "forces", "stress"]
        self.results = {}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        # Standard ASE setup
        if properties is None:
            properties = ["energy"]
        super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]

        # Simulation Logic
        # Read from parameters if present (reconstructed via Heal)
        mode = self.parameters.get("failure_mode", self.failure_mode)

        if mode == "always_fail":
            msg = "Persistent error"
            raise RuntimeError(msg)
        if mode == "parameter_sensitive":
            # Fail if mixing_beta is high (default 0.7 in config)
            # Succeed if mixing_beta is low (0.3)
            beta = self.parameters.get("mixing_beta", 0.7)
            if beta > 0.5:
                msg = "Convergence failed (beta too high)"
                raise RuntimeError(msg)

        # Success results
        n_atoms = len(self.atoms) if self.atoms else 0
        self.results = {
            "energy": -100.0,
            "forces": np.zeros((n_atoms, 3)),
            "stress": np.zeros((3, 3)),  # Full tensor as expected by QEOracle check
        }


@pytest.fixture
def qe_config() -> QEOracleConfig:
    return QEOracleConfig(
        name=OracleType.QE,
        kspacing=0.1,
        mixing_beta=0.7,  # Default high
        ecutwfc=30.0,
        ecutrho=150.0,
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
    mock_espresso_cls: Any, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Use config with low beta to succeed immediately
    qe_config.mixing_beta = HEALER_MIXING_BETA_TARGET
    mock_espresso_cls.side_effect = lambda **kwargs: FakeEspresso(
        failure_mode="parameter_sensitive", **kwargs
    )

    oracle = QEOracle(qe_config)
    results = list(oracle.compute([structure]))

    assert len(results) == 1
    s = results[0]
    assert s.energy == -100.0
    assert s.forces is not None
    assert np.allclose(s.forces, np.zeros((2, 3)))


@patch("mlip_autopipec.components.oracle.qe.Espresso")
def test_qe_compute_healing_success(
    mock_espresso_cls: Any, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Config has beta=0.7, so FakeEspresso fails first.
    # Healer reduces beta to 0.3 and returns NEW calculator instance.
    # Next call to calculate succeeds using the new calculator.

    fake_instance = FakeEspresso(failure_mode="parameter_sensitive", mixing_beta=0.7)
    mock_espresso_cls.return_value = fake_instance

    oracle = QEOracle(qe_config)
    results = list(oracle.compute([structure]))

    assert len(results) == 1
    s = results[0]
    assert s.energy == -100.0

    # Verify provenance
    assert s.tags["qe_params"]["mixing_beta"] == HEALER_MIXING_BETA_TARGET

    # Verify original instance was untouched
    assert fake_instance.parameters["mixing_beta"] == 0.7


@patch("mlip_autopipec.components.oracle.qe.Espresso")
def test_qe_compute_healing_failure(
    mock_espresso_cls: Any, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Setup fake calculator to always fail
    fake_instance = FakeEspresso(failure_mode="always_fail", mixing_beta=0.7)
    mock_espresso_cls.return_value = fake_instance

    oracle = QEOracle(qe_config)
    results = list(oracle.compute([structure]))

    # Should be empty because structure is skipped
    assert len(results) == 0


def test_qe_calculator_setup(qe_config: QEOracleConfig) -> None:
    """Test QECalculator internal logic."""
    with patch("mlip_autopipec.components.oracle.qe.Espresso") as mock_espresso_cls:
        qe_calc = QECalculator(qe_config)
        atoms = Atoms("H")
        qe_calc.calculate(atoms)

        mock_espresso_cls.assert_called()
        call_kwargs = mock_espresso_cls.call_args[1]
        assert call_kwargs["ecutwfc"] == 30.0
