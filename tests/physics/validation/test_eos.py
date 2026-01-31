import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.physics.validation.eos import EOSValidator


class MockCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy"], system_changes=None):
        super().calculate(atoms, properties, system_changes)
        # Simple parabola E = k * (V - V0)^2
        vol = atoms.get_volume()
        v0 = 64.0  # Matches the input structure volume approx
        k = 0.01
        self.results["energy"] = k * (vol - v0) ** 2
        self.results["forces"] = np.zeros((len(atoms), 3))


def test_eos_validation_success(tmp_path):
    """Test that EOS validation passes for a perfect curve."""
    structure = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        pbc=True,
    )

    config = ValidationConfig(eos_vol_range=0.1, eos_n_points=10)
    calc = MockCalculator()

    validator = EOSValidator(structure, calc, config, tmp_path, "test_pot")
    result = validator.validate()

    assert result.overall_status == "PASS"
    assert len(result.metrics) == 1
    assert result.metrics[0].name == "Bulk Modulus (EOS)"
    # For a parabola (V-V0)^2, second derivative is positive, so B > 0
    assert result.metrics[0].value > 0
    assert (tmp_path / "eos_plot.png").exists()


def test_eos_validation_failure(tmp_path):
    """Test that EOS validation fails for inverted curve (unstable)."""
    structure = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        pbc=True,
    )

    class BadCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=["energy"], system_changes=None):
            super().calculate(atoms, properties, system_changes)
            vol = atoms.get_volume()
            v0 = 64.0
            # Inverted parabola: E = -k * (V - V0)^2 -> Unstable
            k = 0.01
            self.results["energy"] = -k * (vol - v0) ** 2
            self.results["forces"] = np.zeros((len(atoms), 3))

    config = ValidationConfig(eos_vol_range=0.1, eos_n_points=10)
    calc = BadCalculator()

    validator = EOSValidator(structure, calc, config, tmp_path, "test_pot")
    result = validator.validate()

    assert result.overall_status == "FAIL"
    assert result.metrics[0].passed is False
