"""MACE Manager module."""

from typing import Any

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import MaceConfig
from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.utils import validate_structure_integrity_atoms

try:
    from mace.calculators import MACECalculator

    HAS_MACE = True
except ImportError:
    HAS_MACE = False
    MACECalculator = Any


class MaceManager:
    """Manages MACE calculations."""

    def __init__(self, config: MaceConfig) -> None:
        """Initialize the MACE Manager."""
        self.config = config
        self.logger = logger.bind(name="MaceManager")
        self.calculator: Any = None

        if not HAS_MACE:
            self.logger.warning("MACE not installed. Only Mock mode will work.")

    def load_model(self) -> None:
        """Load the MACE model."""
        if not HAS_MACE:
            msg = "MACE is not installed. Cannot load model."
            raise OracleError(msg)

        self.logger.info(f"Loading MACE model from {self.config.model_path}")
        try:
            self.calculator = MACECalculator(
                model_paths=self.config.model_path,
                device=self.config.device,
                default_dtype=self.config.default_dtype,
            )
            self.logger.success("MACE model loaded successfully")
        except Exception as e:
            msg = f"Failed to load MACE model: {e}"
            self.logger.exception(msg)
            raise OracleError(msg) from e

    def compute(self, structure: Atoms) -> Atoms:
        """Run MACE prediction for a single structure."""
        # Validate structure first
        try:
            validate_structure_integrity_atoms(structure)
        except (ValueError, TypeError) as e:
            msg = f"Invalid structure input: {e}"
            raise OracleError(msg) from e

        if self.calculator is None:
            self.load_model()

        # Copy structure to avoid side effects
        calc_structure = structure.copy()  # type: ignore[no-untyped-call]
        if not isinstance(calc_structure, Atoms):
            msg = "Failed to copy structure"
            raise OracleError(msg)

        calc_structure.calc = self.calculator

        try:
            # Trigger calculation
            # ASE's get_potential_energy is untyped, but we expect it to exist
            if not hasattr(calc_structure, "get_potential_energy"):
                msg = "Structure object missing get_potential_energy method"
                raise TypeError(msg)  # noqa: TRY301

            calc_structure.get_potential_energy()  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = f"MACE prediction failed: {e}"
            self.logger.exception(msg)
            raise OracleError(msg) from e
        else:
            return calc_structure
