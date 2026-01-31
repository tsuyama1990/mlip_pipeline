from pathlib import Path
from unittest.mock import MagicMock, patch

from mlip_autopipec.domain_models.config import (
    BulkStructureGenConfig,
    Config,
    LammpsConfig,
    MDConfig,
    PotentialConfig,
)
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.physics.validation.runner import ValidationRunner


def test_validation_runner_integrates_results(tmp_path):
    """Test that runner collects results from all validators."""
    # Setup Config
    config = Config(
        project_name="Test",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        structure_gen=BulkStructureGenConfig(
            element="Si", crystal_structure="diamond", lattice_constant=5.43
        ),
        md=MDConfig(temperature=300, n_steps=100, timestep=0.001, ensemble="NVT"),
        lammps=LammpsConfig(command="echo"),
    )

    # Mock validators
    with (
        patch("mlip_autopipec.physics.validation.runner.StructureGenFactory") as MockFactory,
        patch("mlip_autopipec.physics.validation.runner.EOSValidator") as MockEOS,
        patch(
            "mlip_autopipec.physics.validation.runner.ElasticityValidator"
        ) as MockElastic,
        patch("mlip_autopipec.physics.validation.runner.PhononValidator") as MockPhonon,
        patch("mlip_autopipec.physics.validation.runner.LAMMPS"),
        patch("mlip_autopipec.physics.validation.runner.ReportGenerator") as MockReport,
    ):
        # Mock Structure Gen
        mock_gen = MagicMock()
        mock_struct = MagicMock()
        mock_struct.to_ase.return_value = MagicMock()  # ASE Atoms
        mock_gen.generate.return_value = iter([mock_struct])
        MockFactory.create.return_value = mock_gen

        # Mock Validators results
        MockEOS.return_value.validate.return_value = ValidationResult(
            potential_id="pot",
            metrics=[ValidationMetric(name="EOS", value=1, passed=True)],
            plots={},
            overall_status="PASS",
        )
        MockElastic.return_value.validate.return_value = ValidationResult(
            potential_id="pot",
            metrics=[ValidationMetric(name="Elastic", value=1, passed=True)],
            plots={},
            overall_status="PASS",
        )
        MockPhonon.return_value.validate.return_value = ValidationResult(
            potential_id="pot",
            metrics=[ValidationMetric(name="Phonon", value=1, passed=True)],
            plots={},
            overall_status="PASS",
        )

        # Mock Report
        MockReport.return_value.generate.return_value = tmp_path / "report.html"
        (tmp_path / "report.html").write_text("Report Content")

        runner = ValidationRunner(tmp_path)
        result = runner.validate(Path("pot.yace"), config)

        assert result.overall_status == "PASS"
        assert len(result.metrics) == 3
        # Check if report was copied to root (mocking file write would be safer but checking existence is ok)
        assert Path("validation_report.html").exists()

        # Cleanup
        Path("validation_report.html").unlink()


def test_validation_runner_missing_phonopy(tmp_path):
    """Test runner behavior when Phonopy is missing."""
    config = Config(
        project_name="Test",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        structure_gen=BulkStructureGenConfig(
            element="Si", crystal_structure="diamond", lattice_constant=5.43
        ),
        md=MDConfig(temperature=300, n_steps=100, timestep=0.001, ensemble="NVT"),
        lammps=LammpsConfig(command="echo"),
    )

    with (
        patch("mlip_autopipec.physics.validation.runner.StructureGenFactory") as MockFactory,
        patch("mlip_autopipec.physics.validation.runner.EOSValidator") as MockEOS,
        patch(
            "mlip_autopipec.physics.validation.runner.ElasticityValidator"
        ) as MockElastic,
        patch("mlip_autopipec.physics.validation.runner.PhononValidator") as MockPhonon,
        patch("mlip_autopipec.physics.validation.runner.LAMMPS"),
        patch("mlip_autopipec.physics.validation.runner.ReportGenerator") as MockReport,
    ):
        mock_gen = MagicMock()
        mock_struct = MagicMock()
        mock_struct.to_ase.return_value = MagicMock()
        mock_gen.generate.return_value = iter([mock_struct])
        MockFactory.create.return_value = mock_gen

        MockEOS.return_value.validate.return_value = ValidationResult(
            potential_id="pot", metrics=[], plots={}, overall_status="PASS"
        )
        MockElastic.return_value.validate.return_value = ValidationResult(
            potential_id="pot", metrics=[], plots={}, overall_status="PASS"
        )

        # PhononValidator raises RuntimeError on init
        MockPhonon.side_effect = RuntimeError("Phonopy not installed")

        MockReport.return_value.generate.return_value = tmp_path / "report.html"
        (tmp_path / "report.html").write_text("Report Content")

        runner = ValidationRunner(tmp_path)
        result = runner.validate(Path("pot.yace"), config)

        # Should be FAIL because missing Phonopy
        assert result.overall_status == "FAIL"

        # Check metrics contain error
        phonon_metric = next(m for m in result.metrics if m.name == "Phonon")
        assert phonon_metric.passed is False
        assert "Phonopy not installed" in phonon_metric.error_message
