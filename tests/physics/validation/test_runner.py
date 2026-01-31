from unittest.mock import MagicMock
from pathlib import Path

from mlip_autopipec.physics.validation.runner import ValidationRunner
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult

def test_validation_runner():
    # Setup
    val_config = ValidationConfig()
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
    runner = ValidationRunner(val_config, pot_config, potential_path=Path("dummy.yace"))

    # Mock validators
    metric_pass = ValidationMetric(name="Test1", value=1.0, passed=True)
    metric_fail = ValidationMetric(name="Test2", value=-1.0, passed=False)

    runner.eos_validator.validate = MagicMock(return_value=metric_pass)
    runner.elastic_validator.validate = MagicMock(return_value=metric_pass)
    # Phonon returns tuple (metric, plot_path)
    runner.phonon_validator.validate = MagicMock(return_value=(metric_fail, Path("plot.png")))

    runner.report_generator.generate = MagicMock()

    # Execute
    # Mock structure
    from ase import Atoms
    structure = Structure.from_ase(Atoms("Si"))

    result = runner.validate(structure)

    # Verify
    assert isinstance(result, ValidationResult)
    assert result.overall_status == "FAIL" # One failure
    assert len(result.metrics) == 3
    assert "Test2" in [m.name for m in result.metrics]
    assert result.plots["Phonon Dispersion"] == Path("plot.png")
