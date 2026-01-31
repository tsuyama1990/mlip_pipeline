from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.reporting.html_gen import ReportGenerator


class BaseValidator(ABC):
    def __init__(self, config: ValidationConfig, potential_config: PotentialConfig, lammps_command: str = "lammps"):
        self.config = config
        self.potential_config = potential_config
        self.lammps_command = lammps_command

    @abstractmethod
    def validate(self, potential_path: Path) -> ValidationResult:
        pass


class ValidationRunner:
    def __init__(self, config: ValidationConfig, potential_config: PotentialConfig, lammps_command: str = "lammps"):
        self.config = config
        self.potential_config = potential_config
        self.lammps_command = lammps_command

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_generator = ReportGenerator(output_dir=self.config.output_dir)

    def validate(self, potential_path: Path) -> ValidationResult:
        from mlip_autopipec.physics.validation.eos import EOSValidator
        from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
        from mlip_autopipec.physics.validation.phonon import PhononValidator

        validators: list[BaseValidator] = [
            EOSValidator(self.config, self.potential_config, self.lammps_command),
            ElasticityValidator(self.config, self.potential_config, self.lammps_command),
            PhononValidator(self.config, self.potential_config, self.lammps_command),
        ]

        metrics: list[ValidationMetric] = []
        plots: dict[str, Path] = {}
        statuses: list[str] = []

        for validator in validators:
            result = validator.validate(potential_path)
            metrics.extend(result.metrics)
            plots.update(result.plots)
            statuses.append(result.overall_status)

        overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"
        if "FAIL" in statuses:
            overall_status = "FAIL"
        elif "WARN" in statuses:
            overall_status = "WARN"

        result = ValidationResult(
            potential_id=potential_path.stem,
            metrics=metrics,
            plots=plots,
            overall_status=overall_status,
        )

        self.report_generator.generate(result)

        return result
