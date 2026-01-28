import logging
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.validation import PhononConfig
from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult

logger = logging.getLogger(__name__)


class PhononValidator:
    """
    Validates Phonon spectra/frequencies.
    """

    def __init__(self, config: PhononConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, atoms: Atoms, potential_path: Path) -> ValidationResult:
        logger.info("Starting Phonon Validation...")
        try:
            # Logic for phonon calculation...

            metric = ValidationMetric(
                name="phonon_frequencies",
                value=0.0,
                unit="THz",
                passed=False,
                details={"reference": 0.0}
            )

            return ValidationResult(
                module="phonon",
                passed=False,
                metrics=[metric],
                error=None
            )
        except Exception as e:
            return ValidationResult(
                module="phonon",
                passed=False,
                error=str(e),
                metrics=[]
            )
