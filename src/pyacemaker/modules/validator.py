"""Validator module implementation."""

from collections.abc import Iterable

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.interfaces import Validator
from pyacemaker.domain_models.models import Potential, StructureMetadata


class MockValidator(Validator):
    """Validator implementation."""

    def run(self) -> ModuleResult:
        """Run the validator."""
        self.logger.info("Running Validator")
        return ModuleResult(status="success")

    def validate(self, potential: Potential, test_set: Iterable[StructureMetadata]) -> ModuleResult:
        """Validate potential."""
        # Simple implementation that checks if we have enough data and returns mock metrics
        # For mock validation, we consume the iterable.
        valid_set = [s for s in test_set if s.energy is not None]

        if not valid_set:
            self.logger.warning(
                "No valid structures in test set (with energy). Skipping validation."
            )
            # Return failure or skip status
            return ModuleResult(
                status="skipped", metrics=Metrics.model_validate({"count": 0}), artifacts={}
            )

        self.logger.info(f"Validating {potential.path} on {len(valid_set)} valid structures (mock)")

        # Mock metrics
        metrics = {"rmse_energy": 0.005, "rmse_forces": 0.05}

        # Check against configured thresholds if available
        # self.config.validator.thresholds ...
        thresholds = getattr(self.config.validator, "thresholds", {})

        status = "success"
        default_threshold = 1.0
        if thresholds:
            if metrics["rmse_energy"] > thresholds.get("rmse_energy", default_threshold):
                status = "failed"
            if metrics["rmse_forces"] > thresholds.get("rmse_forces", default_threshold):
                status = "failed"

        return ModuleResult(
            status=status,
            metrics=Metrics.model_validate(metrics),
            artifacts={"report": "validation_report.html"},
        )
