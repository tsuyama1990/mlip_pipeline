import logging
import re
import tempfile
import uuid
from pathlib import Path

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
        dangerous_patterns = [
            r";", r"\|", r"&&", r"\$", r"`", r">", r"<", r"\\", r"!", r"\("
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                msg = f"Security violation: Command contains dangerous pattern '{pattern}'"
                raise ValueError(msg)

        # Ensure the command looks like a valid executable invocation
        # Should not start with '-'
        if command.strip().startswith("-"):
             msg = "Security violation: Command cannot start with a flag"
             raise ValueError(msg)

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

            # Use a shared temporary directory base or per-structure?
            # Creating one per structure ensures clean state but adds overhead.
            # Let's use one per structure as per plan/logic.

            for atoms in structures:
                # Calculate K-points
                try:
                    # kpts2mp converts density (kspacing) to grid (nk1, nk2, nk3)
                    # kspacing is in 1/Angstrom (e.g., 0.04)
                    kgrid = kpts2mp(atoms, kpts=self.config.kspacing)
                except Exception as e:
                    logger.warning(f"Failed to calculate k-points: {e}. Using default (1,1,1).")
                    kgrid = (1, 1, 1)

                success = False

                # Filter scf_params to prevent overwriting critical keys
                protected_keys = {"command", "pseudopotentials", "pseudo_dir", "tprnfor", "tstress", "kpts"}
                safe_scf_params = {k: v for k, v in self.config.scf_params.items() if k not in protected_keys}

                # Base parameters
                base_params = {
                    "command": self.config.command,
                    "pseudopotentials": self.config.pseudopotentials,
                    "pseudo_dir": str(self.config.pseudo_dir),
                    "tprnfor": True,
                    "tstress": True,
                    "kpts": kgrid,
                    **safe_scf_params
                }

                # Iterate through recovery recipes
                for recipe in self.recovery_strategy.get_recipes():
                    current_params = base_params.copy()
                    current_params.update(recipe)

                    # Create temporary directory for this calculation attempt
                    with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_calc_dir:
                        try:
                            calc = Espresso(
                                directory=tmp_calc_dir,
                                **current_params
                            )
                            atoms.calc = calc

                            # Run calculation
                            energy = atoms.get_potential_energy()
                            atoms.get_forces()  # Updates atoms.arrays['forces']
                            stress = atoms.get_stress()

                            # Store results
                            atoms.info["energy"] = energy
                            atoms.info["stress"] = stress
                            # forces are stored in atoms.arrays automatically by get_forces() usually
                            # but explicit array assignment ensures it persists if calculator is removed?
                            # get_forces() updates the atoms object.

                            success = True
                            break # Success

                        except CalculatorError as ce:
                            logger.warning(f"Calculation failed with params {recipe}: {ce}. Retrying...")
                        except Exception as ex:
                            logger.exception(f"Unexpected error during calculation: {ex}")
                            # Don't retry for unexpected errors (could be system issue)
                            break

                if success:
                    # Append to output file
                    # Ensure format is extxyz to preserve info/arrays
                    write(output_path, atoms, format="extxyz", append=True)
                    count += 1
                else:
                    logger.error("Failed to converge structure after all recovery attempts. Skipping.")

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
