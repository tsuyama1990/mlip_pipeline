"""Validator module implementation."""

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.interfaces import Validator
from pyacemaker.domain_models.models import Potential, StructureMetadata


class MockValidator(Validator):
    """Validator implementation."""

    def run(self) -> ModuleResult:
        """Run the validator."""
        self.logger.info("Running Validator")
        return ModuleResult(status="success")

    def validate(self, potential: Potential, test_set: list[StructureMetadata]) -> ModuleResult:
        """Validate potential."""
        valid_set = [s for s in test_set if s.energy is not None]
        self.logger.info(f"Validating {potential.path} on {len(valid_set)} valid structures (mock)")

        # Check metrics against thresholds (mock)
        metrics = {"rmse_energy": 0.005, "rmse_forces": 0.05}

        status = "success"
        if metrics["rmse_energy"] > 0.01:
            status = "failed"

        return ModuleResult(
            status=status,
            metrics=Metrics.model_validate(metrics),
            artifacts={"report": "validation_report.html"},
        )
