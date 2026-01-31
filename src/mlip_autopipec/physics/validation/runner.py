from pathlib import Path
from typing import Protocol

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import (
    ValidationConfig,
    ValidationMetric,
    ValidationResult,
)
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
from mlip_autopipec.physics.validation.eos import EOSValidator
from mlip_autopipec.physics.validation.phonon import PhononValidator


class Validator(Protocol):
    def validate(
        self, structure: Structure, potential_path: Path
    ) -> tuple[list[ValidationMetric], dict[str, Path]]: ...


class ValidationRunner:
    """
    Orchestrates the physics validation of a potential.
    """

    def __init__(self, config: Config, work_dir: Path = Path("_work_validation")):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.val_config = config.validation or ValidationConfig()

    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Run all validation tests on the given potential.
        """
        # Generate Reference Structure
        gen = StructureGenFactory.get_generator(self.config.structure_gen)
        structure = gen.generate(self.config.structure_gen)

        metrics: list[ValidationMetric] = []
        plots: dict[str, Path] = {}

        # Instantiate sub-validators
        validators: list[Validator] = [
            EOSValidator(self.val_config, self.config.potential, self.work_dir),
            ElasticityValidator(self.val_config, self.config.potential, self.work_dir),
            PhononValidator(self.val_config, self.config.potential, self.work_dir),
        ]

        for v in validators:
            m, p = v.validate(structure, potential_path)
            metrics.extend(m)
            plots.update(p)

        status = "PASS"
        if any(not m.passed for m in metrics):
            status = "FAIL"

        return ValidationResult(
            potential_id=potential_path.name,
            metrics=metrics,
            plots=plots,
            overall_status=status,  # type: ignore[arg-type]
        )
