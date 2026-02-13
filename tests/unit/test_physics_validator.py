from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import (
    GeneratorConfig,
    GlobalConfig,
    OrchestratorConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.validator.eos import EOSResults
from mlip_autopipec.validator.phonon import PhononResults
from mlip_autopipec.validator.physics import PhysicsValidator


@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    # Create dummy seed file
    seed_file = tmp_path / "seed.xyz"
    seed_file.touch()

    config = MagicMock(spec=GlobalConfig)
    config.generator = MagicMock(spec=GeneratorConfig)
    config.generator.seed_structure_path = seed_file
    config.validator = MagicMock(spec=ValidatorConfig)
    config.validator.strain_magnitude = 0.01
    config.validator.phonon_stability = True
    config.validator.phonon_supercell = [2, 2, 2]
    config.orchestrator = MagicMock(spec=OrchestratorConfig)
    config.orchestrator.work_dir = tmp_path / "work"
    config.orchestrator.max_cycles = 1
    return config

def test_physics_validator_success(mock_config: GlobalConfig, tmp_path: Path) -> None:
    # Setup mocks
    with patch("mlip_autopipec.validator.physics.EOSAnalyzer"), \
         patch("mlip_autopipec.validator.physics.ElasticAnalyzer") as MockElastic, \
         patch("mlip_autopipec.validator.physics.PhononAnalyzer") as MockPhonon, \
         patch("mlip_autopipec.validator.physics.ReportGenerator") as MockReport, \
         patch("mlip_autopipec.validator.physics.fit_birch_murnaghan") as mock_fit_eos, \
         patch("mlip_autopipec.validator.physics.MLIPCalculatorFactory"), \
         patch("ase.io.read") as mock_read:

         # Mock structure read - return real Atoms
         atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
         mock_read.return_value = [atoms]

         validator = PhysicsValidator(mock_config)

         # Mock Analyzer Results
         mock_fit_eos.return_value = EOSResults(
             volume=10.0, energy=-5.0, bulk_modulus=100.0, bulk_modulus_derivative=4.0
         )

         # ElasticAnalyzer returns dict
         mock_elastic_instance = MockElastic.return_value
         mock_elastic_instance.analyze.return_value = {
             "C11": 100.0, "C12": 50.0, "C44": 30.0, "bulk_modulus": 66.6, "shear_modulus": 30.0
         }

         mock_phonon_instance = MockPhonon.return_value
         mock_phonon_instance.analyze.return_value = PhononResults(
             max_imaginary_freq=0.0, is_stable=True, band_structure_plot_data={}
         )

         mock_report_instance = MockReport.return_value
         mock_report_instance.generate.return_value = tmp_path / "report.html"

         # Run validation
         potential = Potential(path="dummy.yace", format="yace")
         result = validator.validate(potential)

         assert result.passed is True
         assert result.metrics["bulk_modulus_eos"] == 100.0
         assert result.metrics["C11"] == 100.0
         assert result.metrics["max_imaginary_freq"] == 0.0
         assert result.report_path == tmp_path / "report.html"

def test_physics_validator_failure_born(mock_config: GlobalConfig, tmp_path: Path) -> None:
    # Setup mocks for failure case
    with patch("mlip_autopipec.validator.physics.EOSAnalyzer"), \
         patch("mlip_autopipec.validator.physics.ElasticAnalyzer") as MockElastic, \
         patch("mlip_autopipec.validator.physics.PhononAnalyzer") as MockPhonon, \
         patch("mlip_autopipec.validator.physics.ReportGenerator") as MockReport, \
         patch("mlip_autopipec.validator.physics.fit_birch_murnaghan") as mock_fit_eos, \
         patch("mlip_autopipec.validator.physics.MLIPCalculatorFactory"), \
         patch("ase.io.read") as mock_read:

         atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
         mock_read.return_value = [atoms]

         validator = PhysicsValidator(mock_config)

         mock_report_instance = MockReport.return_value
         mock_report_instance.generate.return_value = tmp_path / "report.html"

         mock_fit_eos.return_value = EOSResults(10, -5, 100, 4)

         # Unstable Elastic (C11 - C12 < 0)
         mock_elastic_instance = MockElastic.return_value
         mock_elastic_instance.analyze.return_value = {
             "C11": 50.0, "C12": 100.0, "C44": 30.0, "bulk_modulus": 83.3, "shear_modulus": 30.0
         }

         mock_phonon_instance = MockPhonon.return_value
         mock_phonon_instance.analyze.return_value = PhononResults(0.0, True, {})

         potential = Potential(path="dummy.yace", format="yace")
         result = validator.validate(potential)

         assert result.passed is False

def test_physics_validator_no_seed(mock_config: GlobalConfig) -> None:
    # Mock config to have no seed
    mock_config.generator.seed_structure_path = None

    validator = PhysicsValidator(mock_config)
    potential = Potential(path="dummy.yace", format="yace")
    result = validator.validate(potential)

    assert result.passed is True
    assert result.metadata["status"] == "skipped_no_seed"
