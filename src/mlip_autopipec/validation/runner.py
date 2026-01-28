from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.validation.elasticity import ElasticityValidator
from mlip_autopipec.validation.eos import EOSValidator
from mlip_autopipec.validation.phonon import PhononValidator
from mlip_autopipec.validation.report import ReportGenerator


class ValidationRunner:
    """
    Orchestrates the execution of various physics validators.
    """

    def __init__(self, config: ValidationConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, atoms: Atoms, potential_path: Path, modules: list[str] | None = None
    ) -> list[ValidationResult]:
        """
        Run selected validation modules.

        Args:
            atoms: The structure to validate.
            potential_path: Path to the potential file to be used for validation.
            modules: List of modules to run ("phonon", "elastic", "eos").

        Returns:
            List of ValidationResult objects.
        """
        if not isinstance(atoms, Atoms):
            msg = f"Expected ase.Atoms object, got {type(atoms)}"
            raise TypeError(msg)

        results = []
        if modules is None:
            modules = ["phonon", "elastic", "eos"]

        # Ensure lowercase
        modules = [m.lower() for m in modules]

        if "phonon" in modules:
            phonon_validator = PhononValidator(
                self.config.phonon, work_dir=self.work_dir / "phonon"
            )
            results.append(phonon_validator.validate(atoms, potential_path))

        if "elastic" in modules:
            elastic_validator = ElasticityValidator(
                self.config.elastic, work_dir=self.work_dir / "elastic"
            )
            results.append(elastic_validator.validate(atoms, potential_path))

        if "eos" in modules:
            eos_validator = EOSValidator(self.config.eos, work_dir=self.work_dir / "eos")
            results.append(eos_validator.validate(atoms, potential_path))

        # Generate Report
        report_generator = ReportGenerator(self.work_dir)
        report_generator.generate(results, potential_path)

        # Check for global failure if configured
        if self.config.fail_on_instability:
            failed_modules = [r.module for r in results if not r.passed]
            if failed_modules:
                msg = f"Validation failed for modules: {failed_modules}"
                raise RuntimeError(msg)

        return results
