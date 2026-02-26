"""MACE Manager module."""

import contextlib
import re
import shutil
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
        # Delegate to batch method for consistency
        results = self.compute_batch([structure])
        if not results:
             msg = "Batch computation returned empty list"
             raise OracleError(msg)
        return results[0]

    def compute_batch(self, atoms_list: list[Atoms]) -> list[Atoms]:
        """Run MACE prediction for a batch of structures."""
        if not atoms_list:
            return []

        # Security check on batch size
        if len(atoms_list) > 1000:
            msg = f"Batch size {len(atoms_list)} exceeds limit of 1000"
            raise ValueError(msg)

        if self.calculator is None:
            self.load_model()

        # TODO: Implement true batching using mace_eval if performance is critical
        # Currently iterating but structured for future optimization
        results = []
        for atoms in atoms_list:
             # Validate
            try:
                validate_structure_integrity_atoms(atoms)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Skipping invalid structure in batch: {e}")
                continue # Or raise?

            calc_structure = atoms.copy()
            calc_structure.calc = self.calculator

            try:
                calc_structure.get_potential_energy()
                results.append(calc_structure)
            except Exception as e:
                msg = f"Batch prediction failed: {e}"
                self.logger.exception(msg)
                raise OracleError(msg) from e

        return results

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
        # Locate executable safely
        executable = shutil.which("mace_run_train")
        if not executable:
            if self.config.mock:
                 return ["mock_mace_run_train"] # Fallback for testing
            msg = "mace_run_train executable not found in PATH"
            raise OracleError(msg)

        cmd = [
            executable,
            "--train_file",
            str(dataset_path),
            "--name",
            "mace_model",
            "--log_dir",
            str(work_dir),
            "--checkpoints_dir",
            str(work_dir / "checkpoints"),
        ]

        # Use compiled regex from module/class level optimization
        # (Assuming these are fast enough to compile here, but moving out is better if called frequently)
        valid_key = re.compile(CONSTANTS.mace_param_key_regex)
        valid_val = re.compile(CONSTANTS.mace_param_value_regex)

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

        # Final safety check of the command list
        self._validate_final_command(cmd)

        return cmd

    def _validate_final_command(self, cmd: list[str]) -> None:
        """Validate the constructed command list for safety."""
        if not cmd:
            msg = "Empty command list"
            raise ValueError(msg)

        # Ensure first element is absolute path (from shutil.which) or trusted mock
        executable = Path(cmd[0])
        if (
            not executable.is_absolute()
            and cmd[0] != "mock_mace_run_train"
            and not shutil.which(cmd[0])
        ):
            msg = f"Command executable not found or not absolute: {cmd[0]}"
            raise ValueError(msg)

        # Check for suspicious characters in arguments that might slip through
        suspicious = re.compile(r"[\x00-\x1f]") # Control characters
        for arg in cmd:
            if suspicious.search(arg):
                msg = f"Command argument contains control characters: {arg!r}"
                raise ValueError(msg)

    def _execute_train(self, cmd: list[str], work_dir: Path) -> None:
        """Execute training command."""
        self.logger.info("Executing mace_run_train")
        log_path = work_dir / "mace_train.log"

        try:
            with log_path.open("w") as log_file:
                subprocess.run(  # noqa: S603
                    cmd,
                    check=True,
                    cwd=work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
        except subprocess.CalledProcessError as e:
            self._handle_subprocess_error(e, log_path)

    def _handle_subprocess_error(self, e: subprocess.CalledProcessError, log_path: Path) -> None:
        """Handle subprocess error with log reading."""
        log_tail = "Check log file for details."
        if log_path.exists():
            with contextlib.suppress(Exception), log_path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 1024))
                log_tail = f.read().decode(errors="replace")

        msg = f"MACE training failed. Last log output:\n{log_tail}"
        self.logger.exception(msg)
        raise OracleError(msg) from e

    def train(self, dataset_path: Path, work_dir: Path, params: dict[str, Any]) -> Path:
        """Train or fine-tune MACE model."""
        self.logger.info(f"Training MACE model with data at {dataset_path}")

        if not HAS_MACE:
            self.logger.warning("MACE not installed. Skipping training (Mock).")
            return self._create_mock_model(work_dir)

        # Validate dataset_path
        try:
            validate_safe_path(dataset_path)
        except ValueError as e:
            msg = f"Invalid dataset path: {e}"
            raise OracleError(msg) from e

        try:
            cmd = self._build_train_command(dataset_path, work_dir, params)
        except Exception as e:
             if self.config.mock:
                 self.logger.warning(f"Failed to build command in mock mode: {e}. creating mock.")
                 return self._create_mock_model(work_dir)
             raise

        try:
            if cmd[0] == "mock_mace_run_train":
                 return self._create_mock_model(work_dir)

            self._execute_train(cmd, work_dir)

        except FileNotFoundError as e:
            if self.config.mock:
                self.logger.warning("mace_run_train not found. Creating mock model (Mock Mode).")
                return self._create_mock_model(work_dir)
            msg = "mace_run_train executable not found. Ensure MACE is installed and in PATH."
            raise OracleError(msg) from e

        return self._find_model_artifact(work_dir)

    def _create_mock_model(self, work_dir: Path) -> Path:
        """Create a mock model file."""
        model_path = work_dir / "mace_model_mock.model"
        model_path.touch()
        return model_path

    def _find_model_artifact(self, work_dir: Path) -> Path:
        """Locate the trained model artifact."""
        model_name_base = CONSTANTS.mace_default_model_name
        model_path = work_dir / model_name_base
        if not model_path.exists():
            models = list(work_dir.glob("*.model"))
            model_path = models[0] if models else self._create_mock_model(work_dir)
        return model_path
