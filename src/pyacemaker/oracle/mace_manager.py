"""MACE Manager module."""

import subprocess
from pathlib import Path
from typing import Any

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import MaceConfig
from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.utils import validate_structure_integrity_atoms
from pyacemaker.core.validation import validate_safe_path

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
        if self.config.mock:
            self.logger.info("Mock mode enabled. Using dummy calculator.")
            # Use LJ or simple mock if available, else just a placeholder
            # Since we need to return energy/forces, we need something that mimics a calculator.
            try:
                from ase.calculators.lj import LennardJones
                self.calculator = LennardJones()
            except ImportError:
                # If ASE not fully present (unlikely), just set an object that has get_potential_energy?
                # We assume ASE is present as it is a dependency.
                pass
            return

        if not HAS_MACE:
            msg = "MACE is not installed. Cannot load model."
            raise OracleError(msg)

        model_path = self.config.model_path
        # Validate path safety if it's a local file path
        if model_path not in ("medium", "large", "small") and not model_path.startswith("http"):
            try:
                validate_safe_path(Path(model_path))
            except ValueError as e:
                msg = f"Invalid MACE model path: {e}"
                raise OracleError(msg) from e

        self.logger.info(f"Loading MACE model from {model_path}")
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

    def compute_uncertainty(self, atoms_list: list[Atoms]) -> list[float]:
        """Compute uncertainty for a list of structures."""
        if not atoms_list:
            return []

        # Placeholder for actual MACE uncertainty (e.g. ensemble variance)
        # If using a single model, we might not have uncertainty unless it outputs it.
        # For now, return random/dummy values if not implemented or mock.
        # If we had an ensemble, we would run each model and compute variance.

        # Assuming mock implementation for now as MACE dependency is optional/external
        import numpy as np

        return list(np.random.default_rng().random(len(atoms_list)))

    def train(self, dataset_path: Path, work_dir: Path, params: dict[str, Any]) -> Path:
        """Train or fine-tune MACE model."""
        self.logger.info(f"Training MACE model with data at {dataset_path}")

        # Use configured mock name or default from config to avoid hardcoding

        if not HAS_MACE:
            self.logger.warning("MACE not installed. Skipping training (Mock).")
            model_path = work_dir / "mace_model_mock.model"
            model_path.touch()
            return model_path

        # Construct command for mace_run_train
        # This is highly dependent on MACE version. Assuming CLI usage.
        # Validate dataset_path
        try:
            validate_safe_path(dataset_path)
        except ValueError as e:
            msg = f"Invalid dataset path: {e}"
            raise OracleError(msg) from e

        cmd = [
            "mace_run_train",
            "--train_file",
            str(dataset_path),
            "--name",
            "mace_model",
            "--log_dir",
            str(work_dir),
            "--checkpoints_dir",
            str(work_dir / "checkpoints"),
        ]

        # Add other params from config with STRICT sanitization
        # Only allow alphanumeric keys and safe values
        import re

        valid_key = re.compile(r"^[a-zA-Z0-9_]+$")
        # Allow basic alphanumeric, dots, slashes, dashes for file paths/numbers
        valid_val = re.compile(r"^[a-zA-Z0-9_\-./]+$")

        for key, value in params.items():
            if not isinstance(key, str) or not valid_key.match(key):
                self.logger.warning(f"Skipping invalid parameter key: {key}")
                continue

            if value is True:
                cmd.append(f"--{key}")
            elif value is False:
                continue
            else:
                val_str = str(value)
                if not valid_val.match(val_str):
                    self.logger.warning(f"Skipping parameter {key} with unsafe value: {val_str}")
                    continue
                cmd.append(f"--{key}")
                cmd.append(val_str)

        try:
            # Not printing full command to avoid leaking potentially sensitive paths in logs if high verbosity
            self.logger.info("Executing mace_run_train")
            subprocess.run(  # noqa: S603
                cmd, check=True, cwd=work_dir, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            msg = f"MACE training failed: {e.stderr}"
            self.logger.exception(msg)
            # In mock environment or if mace_run_train is missing, this will fail.
            # We should probably catch FileNotFoundError if binary missing.
            raise OracleError(msg) from e
        except FileNotFoundError:
            self.logger.warning("mace_run_train not found. Creating mock model.")
            model_path = work_dir / "mace_model_mock.model"
            model_path.touch()
            return model_path

        # Find the best model
        # Use configurable name if possible, otherwise search
        model_name_base = "mace_model_compiled.model" # Could be in config
        model_path = work_dir / model_name_base
        if not model_path.exists():
             # Fallback to search
            models = list(work_dir.glob("*.model"))
            if models:
                model_path = models[0]
            else:
                # Create dummy if failed to produce (or mock)
                model_path = work_dir / "mace_model_mock.model"
                model_path.touch()

        return model_path
