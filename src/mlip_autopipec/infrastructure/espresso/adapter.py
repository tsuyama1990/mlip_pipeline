import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

from ase import Atoms
from ase.calculators.calculator import kpts2mp
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
    """

    def __init__(self, config: OracleConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self._validate_command(self.config.command)

    def _validate_command(self, command: str | None) -> None:
        """
        Validates the command string for security and availability.
        """
        if not command:
            return

        # Double check for forbidden characters
        forbidden_chars = [";", "|", "&", "`", "$(", ">", "<"]
        if any(char in command for char in forbidden_chars):
            msg = f"Command contains forbidden characters: {command}"
            logger.error(msg)
            raise ValueError(msg)

        # Basic check: verify the executable (first word) exists
        parts = command.split()
        if parts:
            executable = parts[0]
            if not shutil.which(executable):
                logger.warning(f"Command executable '{executable}' not found in PATH.")

        # Check for sensitive paths
        sensitive_paths = ["/etc/passwd", "/root", "/var/run"]
        if any(path in command for path in sensitive_paths):
            msg = f"Command contains suspicious paths: {command}"
            logger.error(msg)
            raise ValueError(msg)

    def _calculate_structure(
        self,
        params: dict[str, Any],
        atom: Atoms,
        temp_path: Path,
        base_input_data: dict[str, Any],
    ) -> Atoms:
        """
        Helper to run QE for a single structure with given params.
        """
        # Construct input_data from base and params
        current_input_data = base_input_data.copy()

        # Update input_data with params (overrides)
        if "mixing_beta" in params:
            current_input_data["electrons"]["mixing_beta"] = params["mixing_beta"]
        if "electron_maxstep" in params:
            current_input_data["electrons"]["electron_maxstep"] = params["electron_maxstep"]

        # Calculate k-points from kspacing
        # kpts2mp(atoms, kpts=density) returns (nx, ny, nz)
        k_grid = kpts2mp(atom, kpts=self.config.kspacing, even=True)  # type: ignore[no-untyped-call]

        calc = Espresso(  # type: ignore[no-untyped-call]
            command=self.config.command,
            pseudopotentials=self.config.pseudopotentials,
            pseudo_dir=str(self.config.pseudo_dir),
            input_data=current_input_data,
            kpts=k_grid,
            directory=str(temp_path),
            label=f"calc_{uuid.uuid4().hex}",
        )

        atom.calc = calc
        # Trigger calculation
        atom.get_potential_energy()  # type: ignore[no-untyped-call]
        # Forces and stress are calculated automatically if tprnfor/tstress are True
        return atom

    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels structures using Quantum Espresso with streaming processing.
        """
        logger.info(f"EspressoOracle: Labeling structures from {dataset.file_path}...")

        if not dataset.file_path.exists():
            msg = f"Dataset file {dataset.file_path} does not exist"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if dataset.file_path.stat().st_size == 0:
            logger.warning(f"Dataset file {dataset.file_path} is empty.")
            # Return empty file (or same empty file path?) - better return a new empty file
            labeled_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.xyz"
            labeled_file.touch()
            return Dataset(file_path=labeled_file)

        labeled_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.xyz"
        total_structures = 0
        successful_structures = 0
        buffer: list[Atoms] = []
        batch_size = self.config.batch_size

        # Default input_data for QE
        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tprnfor": True,
                "tstress": True,
                "disk_io": "none",
            },
            "system": {
                "ecutwfc": 40,
                "ecutrho": 160,
            },
            "electrons": {
                "mixing_beta": 0.7,
                "conv_thr": 1.0e-6,
            },
        }

        # Use one temp dir for the session
        with tempfile.TemporaryDirectory(dir=self.work_dir) as temp_dir:
            temp_path = Path(temp_dir)

            try:
                for atom in iread(dataset.file_path, index=":"):
                    total_structures += 1
                    if atom is None:
                        continue
                    # atom is Atoms object here

                    # Bind current atom using default argument in lambda
                    def calc_func(p: dict[str, Any], a: Atoms = atom) -> Atoms:
                        return self._calculate_structure(p, a, temp_path, input_data)

                    recovery = RecoveryStrategy(base_params={})
                    try:
                        labeled_atom = recovery.attempt_calculation(calc_func)
                        buffer.append(labeled_atom)
                        successful_structures += 1
                    except Exception:
                        # Log the exception stack trace
                        logger.exception(f"Failed to label structure {total_structures}")
                        continue

                    if len(buffer) >= batch_size:
                        write(labeled_file, buffer, append=True)
                        buffer.clear()

                # Write remaining
                if buffer:
                    write(labeled_file, buffer, append=True)
                    buffer.clear()

            except Exception as e:
                msg = f"Error processing dataset: {e}"
                logger.exception(msg)  # already includes message in traceback? No, exception(msg) adds msg.
                raise RuntimeError(msg) from e

        logger.info(
            f"EspressoOracle: Labeling complete. Labeled {successful_structures}/{total_structures} structures."
        )
        return Dataset(file_path=labeled_file)
