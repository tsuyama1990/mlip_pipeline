import logging
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.validation import ElasticConfig
from mlip_autopipec.data_models.validation import ValidationResult

logger = logging.getLogger(__name__)


class ElasticityValidator:
    """
    Validates Elastic Constants.
    """

    def __init__(self, config: ElasticConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, atoms: Atoms, potential_path: Path) -> ValidationResult:
        if not isinstance(atoms, Atoms):
             raise TypeError(f"Expected ase.Atoms, got {type(atoms)}")

        logger.info("Starting Elasticity Validation...")
        self._validate_command(self.config.command)

        return ValidationResult(
            metric="C11",
            value=0.0,
            reference=0.0,
            passed=False
        )

    def _validate_command(self, command: str) -> None:
        if any(c in command for c in [";", "|", "&"]):
            raise ValueError("Unsafe command characters detected")
