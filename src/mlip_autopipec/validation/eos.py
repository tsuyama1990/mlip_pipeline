import logging
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.validation import EOSConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult

logger = logging.getLogger(__name__)


class EOSValidator:
    """
    Validates Equation of State (Bulk Modulus, Equilibrium Volume).
    """

    def __init__(self, config: EOSConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, atoms: Atoms, potential_path: Path) -> ValidationResult:
        logger.info("Starting EOS Validation...")

        try:
            self._validate_command(self.config.command)

            # Placeholder for actual calculation
            bulk_modulus = 0.0

            metric = ValidationMetric(
                name="bulk_modulus",
                value=bulk_modulus,
                unit="GPa",
                passed=False, # Placeholder
                details={"reference": 0.0}
            )

            return ValidationResult(
                module="eos",
                passed=False,
                metrics=[metric]
            )

        except Exception as e:
            return ValidationResult(
                module="eos",
                passed=False,
                error=str(e)
            )

    def _validate_command(self, command: str) -> None:
        if any(c in command for c in [";", "|", "&"]):
            raise ValueError("Unsafe command characters detected")

    def _run_calc(self, atoms: Atoms, command: str) -> float:
        return 0.0
