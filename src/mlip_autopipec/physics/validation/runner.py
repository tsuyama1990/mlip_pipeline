import logging
from pathlib import Path
from typing import Literal

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from .elasticity import ElasticityValidator
from .eos import EOSValidator
from .phonon import PhononValidator
from .reporting.html_gen import ReportGenerator

logger = logging.getLogger("mlip_autopipec")


class ValidationRunner:
    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate(self, potential_path: Path, structure: Structure) -> ValidationResult:
        logger.info(f"Starting validation for {potential_path}")

        # Instantiate validators
        eos = EOSValidator(self.config, potential_path)
        elastic = ElasticityValidator(self.config, potential_path)
        phonon = PhononValidator(self.config, potential_path)

        # Run validators
        # Note: We run sequentially for now.
        logger.info("Running EOS validation...")
        r1 = eos.validate(structure)

        logger.info("Running Elasticity validation...")
        r2 = elastic.validate(structure)

        logger.info("Running Phonon validation...")
        r3 = phonon.validate(structure)

        # Merge results
        metrics = r1.metrics + r2.metrics + r3.metrics
        plots = {**r1.plots, **r2.plots, **r3.plots}

        statuses = [r1.overall_status, r2.overall_status, r3.overall_status]
        overall: Literal["PASS", "WARN", "FAIL"] = "PASS"
        if "FAIL" in statuses:
            overall = "FAIL"
        elif "WARN" in statuses:
            overall = "WARN"

        final_result = ValidationResult(
            potential_id=str(potential_path),
            metrics=metrics,
            plots=plots,
            overall_status=overall,
        )

        # Generate Report
        logger.info(f"Generating validation report at {self.config.report_path}")
        try:
            ReportGenerator.generate(final_result, self.config.report_path)
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

        return final_result
