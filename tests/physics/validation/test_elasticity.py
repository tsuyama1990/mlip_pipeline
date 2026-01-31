from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationConfig
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator


@pytest.fixture
def structure() -> Structure:
    atoms = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
        cell=np.eye(3) * 5.43,
        pbc=True,
    )
    return Structure.from_ase(atoms)


@pytest.fixture
def pot_config() -> PotentialConfig:
    return PotentialConfig(elements=["Si"], cutoff=5.0)


@patch("mlip_autopipec.physics.validation.elasticity.get_validation_calculator")
def test_elasticity_validate_success(
    mock_get_calc: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    calc = MagicMock()
    mock_get_calc.return_value = calc

    config = ValidationConfig()
    validator = ElasticityValidator(config, pot_config, work_dir=tmp_path)

    with patch.object(
        ElasticityValidator, "calculate_stiffness_matrix"
    ) as mock_calc_stiff:
        # Return a stable cubic stiffness matrix
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = 100.0
        C[0, 1] = C[0, 2] = C[1, 2] = C[1, 0] = C[2, 0] = C[2, 1] = 50.0
        C[3, 3] = C[4, 4] = C[5, 5] = 30.0
        mock_calc_stiff.return_value = C

        metrics, plots = validator.validate(structure, potential_path=Path("pot.yace"))

    assert any(m.name == "Elastic Stability" and m.passed for m in metrics)


@patch("mlip_autopipec.physics.validation.elasticity.get_validation_calculator")
def test_elasticity_validate_fail(
    mock_get_calc: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    config = ValidationConfig()
    validator = ElasticityValidator(config, pot_config, work_dir=tmp_path)

    with patch.object(
        ElasticityValidator, "calculate_stiffness_matrix"
    ) as mock_calc_stiff:
        # Unstable
        C = -1.0 * np.eye(6)
        mock_calc_stiff.return_value = C

        metrics, plots = validator.validate(structure, potential_path=Path("pot.yace"))

    assert any(m.name == "Elastic Stability" and not m.passed for m in metrics)


@patch("mlip_autopipec.physics.validation.elasticity.get_validation_calculator")
def test_elasticity_calculation_logic(
    mock_get_calc: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    # Test that calculate_stiffness_matrix actually runs and computes derivatives
    calc = MagicMock()
    mock_get_calc.return_value = calc

    # Mock stress to return constant value -> C should be 0
    calc.get_stress.return_value = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    config = ValidationConfig(elastic_strain_mag=0.01)
    validator = ElasticityValidator(config, pot_config, work_dir=tmp_path)

    atoms = structure.to_ase()
    atoms.calc = calc

    C = validator.calculate_stiffness_matrix(atoms)

    # Deriv = (1 - 1) / ... = 0
    assert np.allclose(C, 0.0)
    assert C.shape == (6, 6)
