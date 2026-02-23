"""Validator module implementation."""

from collections.abc import Iterable

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.interfaces import Validator as ValidatorInterface
from pyacemaker.domain_models.models import Potential, StructureMetadata
from pyacemaker.validator.manager import ValidatorManager


class MockValidator(ValidatorInterface):
    """Mock Validator implementation."""

    def run(self) -> ModuleResult:
        """Run the validator."""
        return ModuleResult(status="success")

    def validate(self, potential: Potential, test_set: Iterable[StructureMetadata]) -> ModuleResult:
        """Validate potential."""
        # Validate consumes the test_set stream
        count = sum(1 for _ in test_set)
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate({"mock": 1.0, "count": count}),
            artifacts={},
        )


class Validator(ValidatorInterface):
    """Validator implementation."""

    def run(self) -> ModuleResult:
        """Run the validator."""
        self.logger.info("Running Validator")
        # In this context, run() is usually not called directly or needs args.
        # Just return success.
        return ModuleResult(status="success")

    def validate(self, potential: Potential, test_set: Iterable[StructureMetadata]) -> ModuleResult:
        """Validate potential."""
        # Stream processing to avoid OOM
        self.logger.info(f"Validating {potential.path}")

        reference_structure = None
        min_e_pa = float("inf")
        count = 0

        # Optimization: Single pass scan
        # We stick to full scan for now as it is strictly O(1) memory (1 structure held).
        for s in test_set:
            count += 1
            if s.features.get("atoms"):
                atoms = s.features["atoms"]
                # Select reference structure (lowest energy/atom from DFT/Source)
                # s.energy should be the ground truth energy (from DFT)
                if s.energy is not None and len(atoms) > 0:
                    e_pa = s.energy / len(atoms)
                    if e_pa < min_e_pa:
                        min_e_pa = e_pa
                        reference_structure = atoms
                elif reference_structure is None:
                    # Fallback to first structure seen if no better candidate yet
                    reference_structure = atoms

        if count == 0:
            self.logger.warning("No structures in test set.")
            return ModuleResult(
                status="skipped", metrics=Metrics.model_validate({"count": 0}), artifacts={}
            )

        self.logger.info(f"Validated on {count} structures (streamed)")

        if reference_structure is None:
            self.logger.error("No valid atoms found in test set structures.")
            return ModuleResult(status="failed", metrics=Metrics(), artifacts={})

        # Initialize Manager
        manager = ValidatorManager(self.config.validator)

        # Output directory
        output_dir = self.config.project.root_dir / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run physics validation
        validation_result = manager.validate(
            potential=potential,
            structure=reference_structure,
            output_dir=output_dir,
        )

        # Merge metrics
        metrics_dict = validation_result.metrics.copy()

        # Calculate RMSE (placeholder - future optimization would use batch predictor here)
        metrics_dict["rmse_energy"] = 0.0
        metrics_dict["rmse_forces"] = 0.0
        metrics_dict["count"] = count

        status = "success" if validation_result.passed else "failed"

        return ModuleResult(
            status=status,
            metrics=Metrics.model_validate(metrics_dict),
            artifacts=validation_result.artifacts,
        )
