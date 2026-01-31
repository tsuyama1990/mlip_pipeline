from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.units import GPa

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationConfig
from mlip_autopipec.physics.validation.eos import EOSValidator


@pytest.fixture
def structure() -> Structure:
    atoms = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
        cell=[[2.7, 2.7, 0], [2.7, 0, 2.7], [0, 2.7, 2.7]],
        pbc=True,
    )
    return Structure.from_ase(atoms)


@pytest.fixture
def pot_config() -> PotentialConfig:
    return PotentialConfig(elements=["Si"], cutoff=5.0)


@patch("mlip_autopipec.physics.validation.eos.get_validation_calculator")
def test_eos_validate_success(
    mock_get_calc: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    calc = MagicMock()
    mock_get_calc.return_value = calc

    # E = 0.5 * (V - 20)**2
    def side_effect(atoms: Atoms = None) -> float:  # type: ignore
        # atoms is not passed to side_effect usually?
        # calc.get_potential_energy() is called on atoms.calc.
        # But atoms.get_potential_energy() calls self.calc.get_potential_energy(self).
        # So first arg is atoms.
        # However, mock call args depend on how it's called.
        # calc.get_potential_energy(atoms=atoms)
        return 10.0  # dummy

    calc.get_potential_energy.return_value = -100.0

    config = ValidationConfig()
    validator = EOSValidator(config, pot_config, work_dir=tmp_path)

    with patch("mlip_autopipec.physics.validation.eos.EquationOfState") as MockEOS:
        mock_eos_instance = MockEOS.return_value
        # v0, e0, B
        mock_eos_instance.fit.return_value = (
            20.0,
            -100.0,
            20.0 * GPa,
        )  # Return B in eV/A^3 if possible?
        # Actually our implementation expects B from fit() to be convertible to GPa by dividing by GPa unit.
        # If we return 20 * GPa, then (20*GPa)/GPa = 20.

        mock_eos_instance.plot.return_value = MagicMock()

        metrics, plots = validator.validate(structure, potential_path=Path("pot.yace"))

    assert len(metrics) == 1
    assert metrics[0].name == "Bulk Modulus"
    assert metrics[0].passed is True
    assert "eos_plot" in plots


@patch("mlip_autopipec.physics.validation.eos.get_validation_calculator")
def test_eos_validate_fail(
    mock_get_calc: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    calc = MagicMock()
    mock_get_calc.return_value = calc
    calc.get_potential_energy.return_value = -10.0

    config = ValidationConfig()
    validator = EOSValidator(config, pot_config, work_dir=tmp_path)

    with patch("mlip_autopipec.physics.validation.eos.EquationOfState") as MockEOS:
        # Pass failing B (negative)
        mock_eos_instance = MockEOS.return_value
        mock_eos_instance.fit.return_value = (20.0, -100.0, -5.0 * GPa)

        metrics, plots = validator.validate(structure, potential_path=Path("pot.yace"))

    assert metrics[0].passed is False
