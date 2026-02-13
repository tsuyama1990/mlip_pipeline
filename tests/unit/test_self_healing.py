from typing import Any

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import OracleConfig
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.oracle.self_healing import run_with_healing


class MockFailingCalculator(Calculator):
    def __init__(self, failures_before_success: int = 0) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.implemented_properties = ["energy", "forces"]
        self.failures_remaining = failures_before_success
        self.attempt_count = 0
        self.parameters: dict[str, Any] = {}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        self.attempt_count += 1
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            msg = "SCF convergence failed"
            raise RuntimeError(msg)

        self.results = {"energy": -10.0, "forces": [[0.0, 0.0, 0.0]] * len(atoms if atoms else [])}

    def set(self, **kwargs: Any) -> None:
        self.parameters.update(kwargs)


def test_self_healing_success_first_try() -> None:
    atoms = Atoms("H")
    calc = MockFailingCalculator(failures_before_success=0)
    atoms.calc = calc

    config = OracleConfig(type=OracleType.DFT, mixing_beta=0.7, smearing_width=0.01)

    run_with_healing(atoms, config)

    assert calc.attempt_count == 1
    # Check results are there
    assert atoms.get_potential_energy() == -10.0  # type: ignore[no-untyped-call]


def test_self_healing_recovery() -> None:
    atoms = Atoms("H")
    # Fail once, then succeed
    calc = MockFailingCalculator(failures_before_success=1)
    atoms.calc = calc

    config = OracleConfig(type=OracleType.DFT, mixing_beta=0.7, smearing_width=0.01)

    # Should catch the error and retry
    run_with_healing(atoms, config)

    assert calc.attempt_count == 2
    # Check if parameters were adjusted.
    # Healing logic: reduce beta, increase smearing?
    # Let's assume the implementation reduces beta by half.
    assert calc.parameters["mixing_beta"] < 0.7
    assert atoms.get_potential_energy() == -10.0  # type: ignore[no-untyped-call]


def test_self_healing_failure() -> None:
    atoms = Atoms("H")
    # Fail many times
    calc = MockFailingCalculator(failures_before_success=10)
    atoms.calc = calc

    config = OracleConfig(type=OracleType.DFT, mixing_beta=0.7, smearing_width=0.01)

    # Should eventually raise the error after max retries
    with pytest.raises(RuntimeError, match="SCF convergence failed"):
        run_with_healing(atoms, config, max_retries=3)

    assert calc.attempt_count >= 3
