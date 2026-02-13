from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidatorConfig
from mlip_autopipec.domain_models.enums import ValidatorType
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.workflow import ValidationResult
from mlip_autopipec.validator.elastic import ElasticResults
from mlip_autopipec.validator.eos import EOSResults
from mlip_autopipec.validator.phonon import PhononResults
from mlip_autopipec.validator.physics import PhysicsValidator


def test_physics_validator_success(tmp_path: Path) -> None:
    """Test PhysicsValidator validates successfully."""
    # Setup mocks
    structure_path = tmp_path / "reference.xyz"
    # Create dummy file (ASE read mock)
    structure_path.touch()

    config = ValidatorConfig(
        type=ValidatorType.PHYSICS,
        structure_path=structure_path,
        elastic_tolerance=0.15,
        phonon_stability=True
    )

    potential = MagicMock(spec=Potential)
    potential.path = Path("mock.pot")

    with patch("mlip_autopipec.validator.physics.read") as mock_read, \
         patch("mlip_autopipec.validator.physics.MLIPCalculatorFactory") as MockFactory, \
         patch("mlip_autopipec.validator.physics.EOSAnalyzer") as MockEOS, \
         patch("mlip_autopipec.validator.physics.ElasticAnalyzer") as MockElastic, \
         patch("mlip_autopipec.validator.physics.PhononAnalyzer") as MockPhonopy, \
         patch("mlip_autopipec.validator.physics.ReportGenerator") as MockReport:

        # Mock structure
        mock_atoms = MagicMock(spec=Atoms)
        mock_read.return_value = mock_atoms
        mock_atoms.get_potential_energy.return_value = -3.5
        mock_atoms.get_volume.return_value = 11.0
        mock_atoms.get_cell.return_value = np.eye(3) * 3.6
        mock_atoms.copy.return_value = mock_atoms # Important for copy() call

        # Mock factory
        mock_calc = MagicMock()
        MockFactory.return_value.create.return_value = mock_calc

        # Mock analyzers
        mock_eos = MockEOS.return_value
        mock_eos.fit_birch_murnaghan.return_value = EOSResults(E0=-3.5, V0=11.0, B0=130.0, B0_prime=5.0)

        mock_elastic = MockElastic.return_value
        mock_elastic.calculate_elastic_constants.return_value = ElasticResults(C11=200.0, C12=100.0, C44=50.0, B=133.3, G=60.0)

        mock_phonon = MockPhonopy.return_value
        mock_phonon.calculate_phonons.return_value = PhononResults(is_stable=True, max_imaginary_freq=0.0, band_structure_path=None)

        mock_report = MockReport.return_value
        mock_report.generate_report.return_value = tmp_path / "report.html"

        validator = PhysicsValidator(config, work_dir=tmp_path)
        result = validator.validate(potential)

        assert isinstance(result, ValidationResult)
        assert result.passed is True

        assert result.metrics["B0"] == 130.0
        assert result.metrics["phonon_stable"] == 1.0

def test_physics_validator_failure_phonon(tmp_path: Path) -> None:
    """Test PhysicsValidator fails on unstable phonon."""
    structure_path = tmp_path / "reference.xyz"
    structure_path.touch()

    config = ValidatorConfig(
        type=ValidatorType.PHYSICS,
        structure_path=structure_path,
        phonon_stability=True
    )

    potential = MagicMock(spec=Potential)
    potential.path = Path("mock.pot")

    with patch("mlip_autopipec.validator.physics.read") as mock_read, \
         patch("mlip_autopipec.validator.physics.MLIPCalculatorFactory") as MockFactory, \
         patch("mlip_autopipec.validator.physics.EOSAnalyzer") as MockEOS, \
         patch("mlip_autopipec.validator.physics.ElasticAnalyzer") as MockElastic, \
         patch("mlip_autopipec.validator.physics.PhononAnalyzer") as MockPhonopy, \
         patch("mlip_autopipec.validator.physics.ReportGenerator") as MockReport:

        mock_atoms = MagicMock(spec=Atoms)
        mock_read.return_value = mock_atoms
        mock_atoms.get_potential_energy.return_value = -3.5
        mock_atoms.get_volume.return_value = 11.0
        mock_atoms.get_cell.return_value = np.eye(3) * 3.6
        mock_atoms.copy.return_value = mock_atoms

        MockFactory.return_value.create.return_value = MagicMock()

        MockEOS.return_value.fit_birch_murnaghan.return_value = EOSResults(E0=-3.5, V0=11.0, B0=130.0, B0_prime=5.0)
        MockElastic.return_value.calculate_elastic_constants.return_value = ElasticResults(C11=200.0, C12=100.0, C44=50.0, B=133.3, G=60.0)

        # Unstable phonon
        MockPhonopy.return_value.calculate_phonons.return_value = PhononResults(is_stable=False, max_imaginary_freq=0.5, band_structure_path=None)

        MockReport.return_value.generate_report.return_value = tmp_path / "report.html"

        validator = PhysicsValidator(config, work_dir=tmp_path)
        result = validator.validate(potential)

        assert result.passed is False
        assert result.metrics["phonon_stable"] == 0.0

def test_physics_validator_missing_structure(tmp_path: Path) -> None:
    """Test error when structure path missing."""
    config = ValidatorConfig(
        type=ValidatorType.PHYSICS,
        structure_path=None
    )

    potential = MagicMock(spec=Potential)
    potential.path = Path("mock.pot")

    validator = PhysicsValidator(config, work_dir=tmp_path)

    with pytest.raises(ValueError, match="Structure path not configured"):
        validator.validate(potential)
