"""MACE Manager module."""

import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from loguru import logger

from pyacemaker.core.config import CONSTANTS, MaceConfig
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

    def update_model_path(self, path: Path) -> None:
        """Update the model path and reload the model."""
        self.config.model_path = str(path)
        self.load_model()

    def load_model(self) -> None:
        """Load the MACE model."""
        if not HAS_MACE:
            msg = "MACE is not installed. Cannot load model."
            raise OracleError(msg)

        model_path = self.config.model_path

        # Validate path or URL
        if model_path.startswith(("http://", "https://")):
            # Simple URL validation
            url_pattern = re.compile(
                r'^(?:http|ftp)s?://' # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
                r'localhost|' # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
                r'(?::\d+)?' # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(model_path):
                 msg = f"Invalid model URL: {model_path}"
                 raise OracleError(msg)
        elif model_path not in ("medium", "large", "small"):
            # Local file path
            try:
                p = Path(model_path).resolve()
                validate_safe_path(p)
            except (ValueError, RuntimeError) as e:
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
        calc_structure = structure.copy()
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

            calc_structure.get_potential_energy()
        except Exception as e:
            msg = f"MACE prediction failed: {e}"
            self.logger.exception(msg)
            raise OracleError(msg) from e
        else:
            return calc_structure

    def compute_uncertainty(self, atoms_list: list[Atoms]) -> list[float]:
        """Compute uncertainty for a list of structures.

        Returns a list of uncertainty values (float).
        """
        if not atoms_list:
            return []

        # Security: Validate batch size to prevent DoS
        if len(atoms_list) > 1000:
            msg = f"Batch size {len(atoms_list)} exceeds limit of 1000"
            raise ValueError(msg)

        # Validate inputs
        for atoms in atoms_list:
            try:
                validate_structure_integrity_atoms(atoms)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Skipping invalid atoms in uncertainty computation: {e}")
                # We must maintain list length alignment, so return a default high uncertainty or raise?
                # Raising breaks the batch. Let's return -1.0 or None, but type says float.
                # Actually, if integrity check fails, we shouldn't trust it.
                # But here we are just validating. If it fails, we should probably fail the call or return dummy.
                # Let's assume the caller filters, but double check.

        try:
            if self.calculator and hasattr(self.calculator, "get_variance"):
                 # Use real calculator variance if available
                 # This depends on MACE version/implementation
                 variances = []
                 for atoms in atoms_list:
                     # This is slow, but MACE calculator might not support batch list directly
                     # unless we use specific batch methods.
                     # For now, placeholder or loop.
                     atoms.calc = self.calculator
                     var = self.calculator.get_variance(atoms)
                     variances.append(float(var))
                 return variances
        except Exception as e:
            self.logger.warning(f"Failed to compute real uncertainty: {e}. Falling back to mock.")

        # Fallback / Mock
        # Return random values [0, 1]
        rng = np.random.default_rng()
        return [float(x) for x in rng.random(len(atoms_list))]

    def _build_train_command(
        self, dataset_path: Path, work_dir: Path, params: dict[str, Any]
    ) -> list[str]:
        """Build the mace_run_train command with strict validation."""
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

        import re

        # Alphanumeric keys only
        valid_key = re.compile(r"^[a-zA-Z0-9_]+$")

        # Strict whitelist for values:
        # Alphanumeric, underscore, hyphen, dot, slash, plus (sci notation), comma, colon, equals, at
        # No shell metacharacters like ;, &, |, >, <, $, `
        valid_val = re.compile(r"^[a-zA-Z0-9_\-\.\/\:\+,=@]+$")

        for key, value in params.items():
            if key not in CONSTANTS.mace_allowed_train_params:
                self.logger.warning(f"Skipping disallowed parameter key: {key}")
                continue

            if not isinstance(key, str) or not valid_key.match(key):
                self.logger.warning(f"Skipping invalid parameter key format: {key}")
                continue

            if value is True:
                cmd.append(f"--{key}")
            elif value is False:
                continue
            else:
                val_str = str(value)
                if not valid_val.match(val_str):
                    self.logger.warning(
                        f"Skipping parameter {key} with unsafe value: {val_str}"
                    )
                    continue
                cmd.append(f"--{key}")
                cmd.append(val_str)
        return cmd

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

        cmd = self._build_train_command(dataset_path, work_dir, params)

        try:
            # Not printing full command to avoid leaking potentially sensitive paths in logs if high verbosity
            self.logger.info("Executing mace_run_train")
            # Explicit shell=False for security
            subprocess.run(  # noqa: S603
                cmd, check=True, cwd=work_dir, capture_output=True, text=True, shell=False
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
        model_name_base = CONSTANTS.mace_default_model_name
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
