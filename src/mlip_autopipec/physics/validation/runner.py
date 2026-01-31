from pathlib import Path
from typing import Literal

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.physics.validation.eos import EOSValidator
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
from mlip_autopipec.physics.validation.phonon import PhononValidator
from mlip_autopipec.physics.reporting.html_gen import ReportGenerator

class ValidationRunner:
    def __init__(self, val_config: ValidationConfig, pot_config: PotentialConfig, potential_path: Path):
        self.val_config = val_config
        self.pot_config = pot_config
        self.potential_path = potential_path

        self.eos_validator = EOSValidator(val_config, pot_config, potential_path)
        self.elastic_validator = ElasticityValidator(val_config, pot_config, potential_path)
        self.phonon_validator = PhononValidator(val_config, pot_config, potential_path)

        self.report_generator = ReportGenerator()

    def validate(self, structure: Structure) -> ValidationResult:
        metrics = []
        plots = {}

        # 1. EOS
        metrics.append(self.eos_validator.validate(structure))

        # 2. Elasticity
        metrics.append(self.elastic_validator.validate(structure))

        # 3. Phonon
        metric_ph, plot_path = self.phonon_validator.validate(structure)
        metrics.append(metric_ph)
        if plot_path:
            plots["Phonon Dispersion"] = plot_path

        # Overall Status
        failures = sum(1 for m in metrics if not m.passed)
        status: Literal["PASS", "WARN", "FAIL"] = "FAIL" if failures > 0 else "PASS"

        result = ValidationResult(
            potential_id=self.potential_path.name,
            metrics=metrics,
            plots=plots,
            overall_status=status
        )

        # Generate Report
        self.report_generator.generate(result, Path("validation_report.html"))

        return result
