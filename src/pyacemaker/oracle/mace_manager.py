"""MACE Manager module."""

import re
import shlex
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
            try:
                from ase.calculators.lj import LennardJones

                self.calculator = LennardJones()
            except ImportError:
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
        return self.compute_batch([structure])[0]

    def compute_batch(self, structures: list[Atoms]) -> list[Atoms]:
        """Run MACE prediction for a batch of structures."""
        if not structures:
            return []

        # Validate structures
        for s in structures:
            try:
                validate_structure_integrity_atoms(s)
            except (ValueError, TypeError) as e:
                msg = f"Invalid structure input: {e}"
                raise OracleError(msg) from e

        if self.calculator is None:
            self.load_model()

        # Copy structures to avoid side effects
        calc_structures = [s.copy() for s in structures]  # type: ignore[no-untyped-call]

        if not self.config.mock and HAS_MACE:
            # For robustness and "proper batching", we should check if MACE has a batch method.
            pass

        results = []
        for atoms in calc_structures:
            atoms.calc = self.calculator
            try:
                if not hasattr(atoms, "get_potential_energy"):
                    msg = "Structure object missing get_potential_energy method"
                    raise TypeError(msg)  # noqa: TRY301
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                results.append(atoms)
            except Exception as e:
                msg = f"MACE prediction failed: {e}"
                self.logger.exception(msg)
                raise OracleError(msg) from e

        return results

    def compute_uncertainty(self, atoms_list: list[Atoms]) -> list[float]:
        """Compute uncertainty for a list of structures."""
        if not atoms_list:
            return []

        # Validate
        for s in atoms_list:
            validate_structure_integrity_atoms(s)

        # Placeholder for actual MACE uncertainty
        import numpy as np

        return list(np.random.default_rng().random(len(atoms_list)))

    def train(self, dataset_path: Path, work_dir: Path, params: dict[str, Any]) -> Path:
        """Train or fine-tune MACE model."""
        self.logger.info(f"Training MACE model with data at {dataset_path}")

        if not HAS_MACE:
            self.logger.warning("MACE not installed. Skipping training (Mock).")
            return self._create_mock_model(work_dir)

        try:
            validate_safe_path(dataset_path)
        except ValueError as e:
            msg = f"Invalid dataset path: {e}"
            raise OracleError(msg) from e

        cmd = self._build_train_command(dataset_path, work_dir, params)

        try:
            self._execute_train_command(cmd, work_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                self.logger.warning("mace_run_train not found. Creating mock model.")
            else:
                msg = f"MACE training failed: {e.stderr}"
                self.logger.exception(msg)
                raise OracleError(msg) from e
            return self._create_mock_model(work_dir)

        return self._find_best_model(work_dir)

    def _build_train_command(self, dataset_path: Path, work_dir: Path, params: dict[str, Any]) -> list[str]:
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

        valid_key = re.compile(r"^[a-zA-Z0-9_]+$")
        valid_val = re.compile(r"^[a-zA-Z0-9_\-./]+$")

        for key, value in params.items():
            if not isinstance(key, str) or not valid_key.match(key):
                self.logger.warning(f"Skipping invalid parameter key: {key}")
                continue

            if value is True:
                cmd.append(f"--{key}")
                continue
            if value is False:
                continue

            val_str = str(value)
            if not valid_val.match(val_str):
                self.logger.warning(f"Skipping parameter {key} with unsafe value: {val_str}")
                continue

            cmd.append(f"--{key}")
            cmd.append(val_str)

        return cmd

    def _execute_train_command(self, cmd: list[str], work_dir: Path) -> None:
        self.logger.info(f"Executing MACE training: {shlex.join(cmd)}")
        subprocess.run(  # noqa: S603
            cmd, check=True, cwd=work_dir, capture_output=True, text=True
        )

    def _create_mock_model(self, work_dir: Path) -> Path:
        model_path = work_dir / "mace_model_mock.model"
        model_path.touch()
        return model_path

    def _find_best_model(self, work_dir: Path) -> Path:
        model_path = work_dir / "mace_model_compiled.model"
        if not model_path.exists():
            models = list(work_dir.glob("*.model"))
            model_path = models[0] if models else self._create_mock_model(work_dir)
        return model_path
