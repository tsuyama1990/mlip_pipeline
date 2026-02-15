"""Validator module implementation."""

from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import CONSTANTS
from pyacemaker.core.interfaces import Validator as ValidatorInterface
from pyacemaker.domain_models.models import Potential, StructureMetadata
from pyacemaker.validator.manager import ValidatorManager


class MockValidator(ValidatorInterface):
    """Mock Validator implementation."""

    def run(self) -> ModuleResult:
        """Run the validator."""
        return ModuleResult(status="success")

    def validate(
        self, potential: Potential, test_set: Iterable[StructureMetadata]
    ) -> ModuleResult:
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
        return ModuleResult(status="success")

    def _process_batch(self, batch: list[StructureMetadata]) -> tuple[Any | None, float]:
        """Process a batch of structures to find local minimum energy structure.

        In a real implementation, this would also calculate RMSEs vectorized.
        """
        local_ref = None
        local_min_e_pa = float("inf")

        for s in batch:
            if s.features.get("atoms"):
                atoms = s.features["atoms"]
                if s.energy is not None and len(atoms) > 0:
                    e_pa = s.energy / len(atoms)
                    if e_pa < local_min_e_pa:
                        local_min_e_pa = e_pa
                        local_ref = atoms
                elif local_ref is None:
                    local_ref = atoms
        return local_ref, local_min_e_pa

    def validate(
        self, potential: Potential, test_set: Iterable[StructureMetadata]
    ) -> ModuleResult:
        """Validate potential.

        Uses batched processing to ensure scalability and prepare for vectorized calculations.
        """
        self.logger.info(f"Validating {potential.path}")

        reference_structure = None
        min_e_pa = float("inf")
        count = 0
        batch_size = 1000

        # Create iterator
        it = iter(test_set)

        while True:
            # Batching logic
            batch = list(islice(it, batch_size))
            if not batch:
                break

            count += len(batch)

            # Process batch
            local_ref, local_min = self._process_batch(batch)

            # Reduce
            if local_min < min_e_pa:
                min_e_pa = local_min
                reference_structure = local_ref
            elif reference_structure is None and local_ref is not None:
                reference_structure = local_ref

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
        pot_path = Path(potential.path)

        validation_result = manager.validate(
            potential_path=pot_path,
            structure=reference_structure,
            output_dir=output_dir,
        )

        metrics_dict = validation_result.metrics.copy()
        # Placeholder RMSE
        metrics_dict["rmse_energy"] = 0.0
        metrics_dict["rmse_forces"] = 0.0

        status = "success" if validation_result.passed else "failed"

        return ModuleResult(
            status=status,
            metrics=Metrics.model_validate(metrics_dict),
            artifacts=validation_result.artifacts,
        )
