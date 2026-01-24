from ase import Atoms

from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.data_models.validation import ValidationResult
from mlip_autopipec.validation.elasticity import ElasticityValidator
from mlip_autopipec.validation.eos import EOSValidator
from mlip_autopipec.validation.phonon import PhononValidator


class ValidationRunner:
    """
    Orchestrates the execution of various physics validators.
    """

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def run(self, atoms: Atoms, modules: list[str] | None = None) -> list[ValidationResult]:
        """
        Run selected validation modules.

        Args:
            atoms: The structure to validate (must have calculator attached).
            modules: List of modules to run ("phonon", "elastic", "eos").
                     If None, runs all configured or available?
                     Usually explicit list is better.

        Returns:
            List of ValidationResult objects.
        """
        results = []
        if modules is None:
            modules = ["phonon", "elastic", "eos"]

        # Ensure lowercase
        modules = [m.lower() for m in modules]

        if "phonon" in modules:
            phonon_validator = PhononValidator(self.config.phonon)
            results.append(phonon_validator.validate(atoms))

        if "elastic" in modules:
            elastic_validator = ElasticityValidator(self.config.elastic)
            results.append(elastic_validator.validate(atoms))

        if "eos" in modules:
            eos_validator = EOSValidator(self.config.eos)
            results.append(eos_validator.validate(atoms))

        return results
