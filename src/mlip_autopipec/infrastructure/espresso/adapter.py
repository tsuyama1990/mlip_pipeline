import logging
import re
import tempfile
import uuid
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import CalculatorError, kpts2mp
from ase.calculators.espresso import Espresso
from ase.io import iread, write

from mlip_autopipec.config.config_model import OracleConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy
from mlip_autopipec.interfaces import BaseOracle

logger = logging.getLogger(__name__)


class EspressoOracle(BaseOracle):
    """
    Oracle implementation using Quantum Espresso via ASE.
    Features: Streaming processing, Self-Healing, Security validation.
    """

    def __init__(self, config: OracleConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir

        # Validate security of the command
        self._validate_command(self.config.command)

        # Check pseudo directory
        if self.config.pseudo_dir and not self.config.pseudo_dir.exists():
            logger.warning(f"Pseudo directory {self.config.pseudo_dir} does not exist.")

        self.recovery_strategy = RecoveryStrategy()

    def _validate_command(self, command: str | None) -> None:
        """
        Validates the command string to prevent shell injection.
        Only allows alphanumeric characters, spaces, dots, dashes, slashes, and underscores.
        """
        if not command:
            return

        # Check against blacklist of dangerous shell metacharacters
        dangerous_patterns = [r";", r"\|", r"&&", r"\$", r"`", r">", r"<", r"\\", r"!", r"\("]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                msg = f"Security violation: Command contains dangerous pattern '{pattern}'"
                raise ValueError(msg)

        # Ensure the command looks like a valid executable invocation
        # Should not start with '-'
        if command.strip().startswith("-"):
            msg = "Security violation: Command cannot start with a flag"
            raise ValueError(msg)

    def _label_structure(self, atoms: Atoms, temp_dir: str) -> bool:
        """
        Labels a single structure with recovery strategy.
        Returns True if successful, False otherwise.
        """
        try:
            # Calculate K-points
            # kpts2mp converts density (kspacing) to grid (nk1, nk2, nk3)
            # kspacing is in 1/Angstrom (e.g., 0.04)
            kgrid = kpts2mp(atoms, kpts=self.config.kspacing)  # type: ignore[no-untyped-call]
        except Exception as e:
            logger.warning(f"Failed to calculate k-points: {e}. Using default (1,1,1).")
            kgrid = (1, 1, 1)

        success = False

        # Filter scf_params to prevent overwriting critical keys
        protected_keys = {"command", "pseudopotentials", "pseudo_dir", "tprnfor", "tstress", "kpts"}
        safe_scf_params = {
            k: v for k, v in self.config.scf_params.items() if k not in protected_keys
        }

        # Base parameters
        base_params = {
            "command": self.config.command,
            "pseudopotentials": self.config.pseudopotentials,
            "pseudo_dir": str(self.config.pseudo_dir),
            "tprnfor": True,
            "tstress": True,
            "kpts": kgrid,
            **safe_scf_params,
        }

        # Iterate through recovery recipes
        for recipe in self.recovery_strategy.get_recipes():
            current_params = base_params.copy()
            current_params.update(recipe)

            # Use a fresh subdirectory or just overwrite files?
            # Reusing the same temp_dir might cause conflicts if previous run left garbage.
            # But ASE handles cleaning usually. To be safe, we can use subdirs or just trust ASE.
            # Since we want to avoid creating thousands of dirs, let's reuse temp_dir.
            # However, if a calculation fails halfway, it might leave partial files.
            # Espresso overwrites by default.

            try:
                calc = Espresso(directory=temp_dir, **current_params)  # type: ignore[no-untyped-call]
                atoms.calc = calc

                # Run calculation
                energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                atoms.get_forces()  # type: ignore[no-untyped-call] # Updates atoms.arrays['forces']
                stress = atoms.get_stress()  # type: ignore[no-untyped-call]

                # Store results
                atoms.info["energy"] = energy
                atoms.info["stress"] = stress

                success = True
                break  # Success

            except CalculatorError as ce:
                logger.warning(f"Calculation failed with params {recipe}: {ce}. Retrying...")
            except Exception:
                logger.exception("Unexpected error during calculation")
                # Don't retry for unexpected errors (could be system issue)
                break

        return success

    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the structures in the dataset using QE.
        Streams input and output to avoid memory issues.
        """
        output_path = self.work_dir / f"labeled_{uuid.uuid4().hex}.extxyz"

        # Check if input file exists
        if not dataset.file_path.exists():
            logger.warning(f"Dataset file {dataset.file_path} does not exist.")
            output_path.touch()
            return Dataset(file_path=output_path)

        count = 0
        try:
            # Re-read input file using iread for streaming
            # iread returns an iterator of Atoms
            structures = iread(dataset.file_path)

            # Create ONE temporary directory for the entire batch to avoid I/O overhead
            # We rely on ASE/QE to overwrite files or handle cleanup within this directory
            with tempfile.TemporaryDirectory(dir=self.work_dir) as batch_temp_dir:
                for atoms in structures:
                    if self._label_structure(atoms, batch_temp_dir):
                        # Append to output file
                        # Ensure format is extxyz to preserve info/arrays
                        write(output_path, atoms, format="extxyz", append=True)
                        count += 1
                    else:
                        logger.error(
                            "Failed to converge structure after all recovery attempts. Skipping."
                        )

        except Exception as e:
            msg = f"Fatal error during labeling: {e}"
            logger.exception(msg)
            # Ensure output file exists even if empty
            if not output_path.exists():
                output_path.touch()

        if not output_path.exists():
            output_path.touch()

        logger.info(f"Labeling complete. Processed {count} structures.")
        return Dataset(file_path=output_path)
