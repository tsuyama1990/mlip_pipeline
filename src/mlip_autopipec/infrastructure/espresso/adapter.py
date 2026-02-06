import logging
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import CalculatorError
from ase.calculators.espresso import Espresso
from ase.io import iread, write

from mlip_autopipec.config import OracleConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy
from mlip_autopipec.interfaces import BaseOracle

logger = logging.getLogger(__name__)


class EspressoOracle(BaseOracle):
    def __init__(self, config: OracleConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self._validate_command(self.config.command)

    def _validate_command(self, command: str | None) -> None:
        if not command:
            return

        # 1. Reject illegal characters to prevent complex shell injection
        # Allows alphanumeric, space, dot, dash, underscore, slash
        if not re.match(r"^[\w\s\-\./]+$", command):
            msg = f"Security check failed: Command '{command}' contains illegal characters."
            raise ValueError(msg)

        # 2. Whitelist check for the executable
        # Split command to get the first token (executable)
        tokens = command.split()
        if not tokens:
            return

        executable = tokens[0]
        # Common MPI runners and Espresso binaries
        whitelist = {
            "pw.x",
            "pw_gpu.x",
            "cp.x",
            "mpirun",
            "mpiexec",
            "srun",
            "orterun",
            "/usr/bin/mpirun",
            "/usr/bin/mpiexec",
            "/usr/bin/srun",
        }

        # Check if exact match or ends with whitelisted binary (handle paths)
        is_whitelisted = False
        if executable in whitelist:
            is_whitelisted = True
        else:
            # Check if basename matches
            name = Path(executable).name
            if name in whitelist:
                is_whitelisted = True

        if not is_whitelisted:
            msg = f"Security check failed: Executable '{executable}' is not in the whitelist."
            raise ValueError(msg)

    def label(self, dataset: Dataset) -> Dataset:
        output_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.extxyz"
        logger.info(
            f"EspressoOracle: Labeling structures from {dataset.file_path} to {output_file}"
        )

        if not dataset.file_path.exists() or dataset.file_path.stat().st_size == 0:
            logger.warning(
                f"Input file {dataset.file_path} is missing or empty. Returning empty dataset."
            )
            output_file.touch()
            return Dataset(file_path=output_file)

        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_dir:
            tmp_path = Path(tmp_dir)
            total_processed = 0
            success_count = 0
            batch_buffer: list[Atoms] = []
            batch_size = 10

            try:
                # Type ignore because iread is not fully typed in ASE stubs
                iterator = iread(dataset.file_path)  # type: ignore[no-untyped-call]

                for atoms in iterator:
                    if not isinstance(atoms, Atoms):
                        continue

                    total_processed += 1
                    if self._process_structure(atoms, tmp_path):
                        batch_buffer.append(atoms)
                        success_count += 1

                        if len(batch_buffer) >= batch_size:
                            write(output_file, batch_buffer, append=True)  # type: ignore[no-untyped-call]
                            batch_buffer.clear()

                if batch_buffer:
                    write(output_file, batch_buffer, append=True)  # type: ignore[no-untyped-call]
                    batch_buffer.clear()

            except Exception as e:
                logger.exception(f"Failed to read or process input file {dataset.file_path}")
                msg = f"Oracle failed processing {dataset.file_path}"
                raise RuntimeError(msg) from e

            self._verify_output(output_file, success_count)

            logger.info(
                f"EspressoOracle: Processed {total_processed} structures, labeled {success_count}."
            )

        return Dataset(file_path=output_file)

    def _process_structure(self, atoms: Atoms, tmp_path: Path) -> bool:
        """
        Attempts to calculate energy/forces for a single structure, with recovery retries.
        Returns True if successful, False otherwise.
        """
        recovery = RecoveryStrategy(self.config.recovery_recipes)
        current_params: dict[str, Any] = {}
        retries = 0

        while True:
            try:
                input_data = self.config.scf_params.copy()
                input_data.update(current_params)
                input_data["tprnfor"] = True
                input_data["tstress"] = True

                calc = Espresso(
                    command=self.config.command,
                    pseudopotentials=self.config.pseudopotentials,
                    pseudo_dir=str(self.config.pseudo_dir),
                    kspacing=self.config.kspacing,
                    input_data=input_data,
                    directory=str(tmp_path),
                    tprnfor=True,
                    tstress=True,
                )  # type: ignore[no-untyped-call]

                atoms.calc = calc
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]

            except CalculatorError as e:
                logger.warning(f"SCF failed (Attempt {retries}): {e}")
                recipe = recovery.get_recipe(retries)
                if recipe is None:
                    logger.exception("Failed all recovery attempts. Skipping.")
                    return False

                logger.info(f"Retrying with recovery recipe: {recipe}")
                current_params.update(recipe)
                retries += 1
            except Exception:
                logger.exception("Unexpected error processing structure")
                return False
            else:
                return True

    def _verify_output(self, output_file: Path, expected_count: int) -> None:
        if output_file.exists():
            try:
                verify_count = 0
                if output_file.stat().st_size > 0:
                    for _ in iread(output_file):  # type: ignore[no-untyped-call]
                        verify_count += 1

                if verify_count != expected_count:
                    msg = f"Output file integrity check failed. Expected {expected_count}, got {verify_count}"
                    logger.error(msg)
                    raise RuntimeError(msg)

            except Exception as e:
                logger.exception("Failed to verify output file integrity")
                msg = "Output validation failed"
                raise RuntimeError(msg) from e
        elif expected_count > 0:
            msg = "Output file missing despite successful labeling"
            logger.error(msg)
            raise RuntimeError(msg)
