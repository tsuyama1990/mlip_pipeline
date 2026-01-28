from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.validation.elasticity import ElasticityValidator
from mlip_autopipec.validation.eos import EOSValidator
from mlip_autopipec.validation.phonon import PhononValidator


class ValidationRunner:
    """
    Orchestrates the execution of various physics validators.
    """

    def __init__(self, config: ValidationConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir

    def run(self, atoms: Atoms, potential_path: Path, modules: list[str] | None = None) -> list[ValidationResult]:
        """
        Run selected validation modules.

        Args:
            atoms: The structure to validate (must have calculator attached).
            potential_path: Path to the potential file (needed for some validators that re-initialize calculator).
            modules: List of modules to run ("phonon", "elastic", "eos").
                     If None, runs all configured or available?
                     Usually explicit list is better.

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
            phonon_validator = PhononValidator(self.config.phonon, self.work_dir)
            results.append(phonon_validator.validate(atoms, potential_path))

        if "elastic" in modules:
            elastic_validator = ElasticityValidator(self.config.elastic, self.work_dir)
            results.append(elastic_validator.validate(atoms, potential_path))

        if "eos" in modules:
            eos_validator = EOSValidator(self.config.eos, self.work_dir)
            results.append(eos_validator.validate(atoms, potential_path))

        return results
