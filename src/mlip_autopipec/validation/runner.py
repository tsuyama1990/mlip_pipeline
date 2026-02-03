import logging
from pathlib import Path

from ase.build import bulk

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.validation.metrics import ElasticValidator, PhononValidator
from mlip_autopipec.validation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ValidationRunner:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self.report_generator = ReportGenerator()

    def validate(self, potential_path: Path, work_dir: Path) -> ValidationResult:
        """
        Runs all enabled validation tests.
        """
        logger.info(f"Starting validation for {potential_path}")
        metrics: list[MetricResult] = []

        # Determine structure to test on
        # For now, as per spec/memory, we default to Copper FCC
        structure = bulk("Cu", "fcc", a=3.6)

        if self.config.check_phonons:
            logger.info("Running Phonon Validation")
            metrics.append(
                PhononValidator().validate(potential_path, structure, work_dir)
            )

        if self.config.check_elastic:
            logger.info("Running Elastic Validation")
            metrics.append(
                ElasticValidator().validate(potential_path, structure, work_dir)
            )

        # Aggregate results
        passed = all(m.passed for m in metrics)

        result = ValidationResult(
            passed=passed,
            metrics=metrics,
            reason="All tests passed" if passed else "Some tests failed",
        )

        # Generate Report
        try:
            report_path = self.report_generator.generate(result, work_dir)
            result.report_path = report_path
        except Exception:
            logger.exception("Failed to generate report")

        return result
