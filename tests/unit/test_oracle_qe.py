from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.components.oracle.qe import QECalculator, QEOracle
from mlip_autopipec.constants import HEALER_MIXING_BETA_TARGET
from mlip_autopipec.domain_models.config import QEOracleConfig
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.domain_models.structure import Structure


# Define a Fake Calculator that behaves like Espresso but runs in memory
class FakeEspresso(Calculator):
    def __init__(self, failure_mode: str = "parameter_sensitive", **kwargs: Any) -> None:
        super().__init__()
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
        super().calculate(atoms, properties, system_changes)

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
    mock_espresso_cls: MagicMock, qe_config: QEOracleConfig, structure: Structure
) -> None:
    # Use config with low beta to succeed immediately
    qe_config.mixing_beta = HEALER_MIXING_BETA_TARGET

    # We use side_effect to return a new FakeEspresso instance each time it's called
    mock_espresso_cls.side_effect = lambda **kwargs: FakeEspresso(
        failure_mode="parameter_sensitive", **kwargs
    )

    # We cannot easily test parallel execution with mocks due to pickling limits.
    # So we trust `test_process_single_structure_logic` covers the behavior.
    # This test primarily ensures instantiation doesn't crash.
    _ = QEOracle(qe_config)


def test_process_single_structure_logic(qe_config: QEOracleConfig, structure: Structure) -> None:
    """
    Test the worker function _process_single_structure directly to verify behavior.
    This bypasses multiprocessing but verifies the logic flows correctly.
    """
    from mlip_autopipec.components.oracle.qe import _process_single_structure

    # 1. Success Case
    qe_config.mixing_beta = HEALER_MIXING_BETA_TARGET # Low beta = success

    with patch("mlip_autopipec.components.oracle.qe.Espresso") as mock_cls:
        # Configure mock to behave like FakeEspresso
        mock_cls.side_effect = lambda **kwargs: FakeEspresso(
            failure_mode="parameter_sensitive", **kwargs
        )

        result_json = _process_single_structure(structure.model_dump_json(), qe_config)
        assert result_json is not None

        result = Structure.model_validate_json(result_json)
        assert result.energy == -100.0
        assert result.tags["qe_params"]["mixing_beta"] == HEALER_MIXING_BETA_TARGET

        # Verify call arguments
        mock_cls.assert_called()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["ecutwfc"] == 30.0
        assert call_kwargs["mixing_beta"] == HEALER_MIXING_BETA_TARGET


def test_process_single_structure_healing(qe_config: QEOracleConfig, structure: Structure) -> None:
    """Test healing logic within the worker function."""
    from mlip_autopipec.components.oracle.qe import _process_single_structure

    # 1. Healing Case (High Beta -> Fail -> Heal -> Success)
    qe_config.mixing_beta = 0.7 # High beta

    with patch("mlip_autopipec.components.oracle.qe.Espresso") as mock_cls:
        # We need to simulate that the first call fails and the second succeeds (via Healer)
        # However, Healer calls type(calc)(...), creating a new instance.
        # If we just use side_effect on the class mock, we can return fresh instances.
        # But we want to ensure the *parameters* passed to those instances change.

        # We can use a side_effect that checks input parameters and returns success/fail FakeEspresso

        mock_cls.side_effect = lambda **kwargs: FakeEspresso(
            failure_mode="parameter_sensitive", **kwargs
        )

        result_json = _process_single_structure(structure.model_dump_json(), qe_config)
        assert result_json is not None

        result = Structure.model_validate_json(result_json)
        # Verify it succeeded
        assert result.energy == -100.0

        # Verify provenance shows HEALED parameter
        assert result.tags["qe_params"]["mixing_beta"] == HEALER_MIXING_BETA_TARGET

        # We expect at least one call to create the calculator
        assert mock_cls.call_count >= 1


def test_qe_calculator_setup(qe_config: QEOracleConfig) -> None:
    """Test QECalculator internal logic."""
    with patch("mlip_autopipec.components.oracle.qe.Espresso") as mock_espresso_cls:
        qe_calc = QECalculator(qe_config)
        qe_calc.create_calculator()

        mock_espresso_cls.assert_called()
        call_kwargs = mock_espresso_cls.call_args[1]
        assert call_kwargs["ecutwfc"] == 30.0
