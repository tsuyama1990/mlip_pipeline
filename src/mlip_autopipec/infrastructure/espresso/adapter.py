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
        }

        # Absolute paths whitelist (system dependent but standard)
        # We can be broader: if it is an absolute path, we require it to be a standard system binary?
        # Or simpler: The Executable Name (basename) MUST be in the whitelist.
        # This prevents running /tmp/malicious/pw.x simply because it is named pw.x?
        # Wait, if I upload a malicious script named pw.x to /tmp/pw.x and run it, that's bad.
        # So I should reject paths that are not standard system paths?
        # But user might have custom install in /opt/ or /home/.
        # The feedback says "validate full path, not just basename".
        # This implies we should check if the provided path is "safe".
        # Safe means it is a known binary name.
        # If the user provides a path, we can't easily verify it's safe unless we whitelist the directory too.
        # Strict approach: executable must be one of the whitelist names (searched in PATH),
        # OR it must be a specific absolute path that we consider safe?
        # Let's start with strict whitelist of names.
        # If the user provides a path like /usr/bin/pw.x, we check if the basename is in whitelist.
        # BUT we must ensure the path itself doesn't point to a user-writable dir? Too complex.

        # Auditor feedback: "Validate full path... Use absolute path resolution and stricter pattern matching."
        # If I resolve the path, what do I check against?
        # Maybe I just enforce that the command token DOES NOT contain '/' unless it is one of a specific set of fully qualified paths?
        # E.g. allow 'pw.x' (uses PATH) or '/usr/bin/pw.x' or '/usr/local/bin/pw.x'.
        # Reject '/tmp/pw.x'.

        # Let's implement: Executable must be in whitelist.
        # If it contains a slash, it must start with /usr/bin/ or /usr/local/bin/ or /opt/.
        # That seems reasonable for a "safe" environment?
        # Or even stricter: Only allow standard PATH lookups (no slashes allowed in executable name).

        if "/" in executable:
             # It's a path. Verify it starts with trusted prefixes
             trusted_prefixes = ("/usr/bin/", "/usr/local/bin/", "/opt/")
             if not any(executable.startswith(prefix) for prefix in trusted_prefixes):
                 msg = f"Security check failed: Path '{executable}' is not in a trusted directory ({trusted_prefixes})."
                 raise ValueError(msg)

             # Also check basename is whitelisted
             name = Path(executable).name
             if name not in whitelist:
                 msg = f"Security check failed: Executable name '{name}' is not whitelisted."
                 raise ValueError(msg)
        # Just a name. Must be in whitelist.
        elif executable not in whitelist:
            msg = f"Security check failed: Executable '{executable}' is not in the whitelist."
            raise ValueError(msg)

    def label(self, dataset: Dataset) -> Dataset:  # noqa: C901
        output_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.extxyz"
        logger.info(f"EspressoOracle: Labeling structures from {dataset.file_path} to {output_file}")

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
            batch_size = self.config.batch_size

            try:
                # Type ignore because iread is not fully typed in ASE stubs
                iterator = iread(dataset.file_path)

                for atoms in iterator:
                    if not isinstance(atoms, Atoms):
                        continue

                    total_processed += 1
                    if self._process_structure(atoms, tmp_path):
                        batch_buffer.append(atoms)
                        success_count += 1

                        if len(batch_buffer) >= batch_size:
                            # Pass a copy of the buffer to write
                            write(output_file, list(batch_buffer), append=True)
                            batch_buffer.clear()

                if batch_buffer:
                    write(output_file, list(batch_buffer), append=True)
                    batch_buffer.clear()

            except Exception as e:
                logger.exception(f"Failed to read or process input file {dataset.file_path}")
                msg = f"Oracle failed processing {dataset.file_path}"
                raise RuntimeError(msg) from e

            # Verify output integrity (without reading whole file)
            # We trust success_count and check basic file properties
            if success_count > 0:
                if not output_file.exists():
                     msg = "Output file missing despite successful labeling"
                     logger.error(msg)
                     raise RuntimeError(msg)
                if output_file.stat().st_size == 0:
                     msg = "Output file is empty despite successful labeling"
                     logger.error(msg)
                     raise RuntimeError(msg)

            logger.info(f"EspressoOracle: Processed {total_processed} structures, labeled {success_count}.")

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
                ) # type: ignore[no-untyped-call]

                atoms.calc = calc
                atoms.get_potential_energy() # type: ignore[no-untyped-call]

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
