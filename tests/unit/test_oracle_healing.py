from typing import Any

import pytest
from ase.calculators.calculator import Calculator

from mlip_autopipec.components.oracle.healing import Healer, HealingFailedError


class MockEspresso(Calculator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.parameters = kwargs


def test_healer_reduce_mixing_beta() -> None:
    """Test healing strategy 1: Reduce mixing_beta."""
    calc = MockEspresso(mixing_beta=0.7)
    healer = Healer()

    new_calc = healer.heal(calc, Exception("Convergence failed"))

    assert new_calc.parameters["mixing_beta"] == 0.3
    # Ensure new instance
    assert new_calc is not calc


def test_healer_increase_smearing() -> None:
    """Test healing strategy 2: Increase smearing (degauss)."""
    # Assuming mixing_beta is already low
    calc = MockEspresso(mixing_beta=0.3, degauss=0.01)
    healer = Healer()

    new_calc = healer.heal(calc, Exception("Convergence failed"))

    # Should not change mixing_beta
    assert new_calc.parameters["mixing_beta"] == 0.3
    # Should increase degauss
    assert new_calc.parameters["degauss"] > 0.01
    assert new_calc.parameters["degauss"] == 0.02  # Assuming double or set value


def test_healer_change_algo() -> None:
    """Test healing strategy 3: Change diagonalization algorithm."""
    # Assuming mixing_beta is low and degauss is high enough
    calc = MockEspresso(mixing_beta=0.3, degauss=0.02, diagonalization="david")
    healer = Healer()

    new_calc = healer.heal(calc, Exception("Convergence failed"))

    assert new_calc.parameters["diagonalization"] == "cg"


def test_healer_exhausted() -> None:
    """Test failure when all strategies exhausted."""
    calc = MockEspresso(mixing_beta=0.3, degauss=0.02, diagonalization="cg")
    healer = Healer()

    with pytest.raises(HealingFailedError):
        healer.heal(calc, Exception("Convergence failed"))


def test_healer_invalid_calculator() -> None:
    """Test healer with non-Espresso calculator (generic)."""

    # Should try best effort or fail if parameters don't match
    # Use simple class that accepts kwargs
    class SimpleCalc(Calculator):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()  # type: ignore[no-untyped-call]
            self.parameters = kwargs

    calc = SimpleCalc()
    healer = Healer()

    # It should probably try setting mixing_beta if not present
    new_calc = healer.heal(calc, Exception("Error"))
    assert new_calc.parameters["mixing_beta"] == 0.3


def test_healer_no_parameters() -> None:
    """Test calculator without parameters attribute."""

    class NoParamCalc:
        pass

    calc = NoParamCalc()
    healer = Healer()
    with pytest.raises(HealingFailedError, match="does not have parameters"):
        healer.heal(calc, Exception("Error"))  # type: ignore[arg-type]
