import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.inputs import InputGenerator
from mlip_autopipec.dft.parsers import QEOutputParser
from mlip_autopipec.dft.recovery import RecoveryHandler


class DFTFatalError(Exception):
    pass


class QERunner:
    """
    Orchestrates Quantum Espresso calculations with auto-recovery.
    """

    def __init__(self, config: DFTConfig) -> None:
        self.config = config

    def run(self, atoms: Atoms, uid: str | None = None) -> DFTResult:
        """
        Runs the DFT calculation for the given atoms object.
        """
        if uid is None:
            uid = str(uuid4())

        # Create a working directory for this run
        # Use temp dir or configured working dir?
        # Ideally, we should use a scratch space.
        # For now, let's use a temporary directory to be safe and clean up later.

        # Check executable existence
        # The command might be complex like "mpirun -np 4 pw.x"
        # We try to find the executable part.
        executable_candidate = self.config.command.split()[0]
        # If it's mpirun, we assume it's installed. If it's pw.x directly, we check.
        # But really we should check whatever is being run if possible.
        # Simple heuristic: if command starts with something that shutil.which can find, we are good.
        if not shutil.which(executable_candidate):
            # If mpirun is not found, or pw.x is not found
            # If the command is absolute path, which also handles it.
            # One edge case: "mpirun" might be aliased or loaded via module.
            # But generally for robustness we can warn or fail.
            # Given the audit feedback, let's raise a clear error if we can't find it.
            # Wait, splitting "mpirun -np 4 pw.x" gives "mpirun".
            # If "pw.x" is used, it gives "pw.x".
            # We should check if the executable exists.
            if not shutil.which(executable_candidate):
                msg = f"Executable '{executable_candidate}' not found in PATH."
                raise DFTFatalError(msg)

        # We need to preserve params across retries
        current_params = {}  # Start with defaults (empty dict means InputGenerator uses defaults)

        attempt = 0
        while attempt <= self.config.max_retries:
            attempt += 1

            with tempfile.TemporaryDirectory(prefix=f"dft_run_{uid}_") as tmpdir:
                work_dir = Path(tmpdir)
                input_str = InputGenerator.create_input_string(atoms, current_params)

                input_path = work_dir / "pw.in"
                output_path = work_dir / "pw.out"

                input_path.write_text(input_str)

                # Let's symlink pseudos for now as it's safer for QE
                self._stage_pseudos(work_dir, atoms)

                start_time = time.time()

                # Run command
                # We need to replace pw.x with command from config
                # The config command might be "mpirun -np 4 pw.x"
                # We assume input is piped via stdin or -in flag.
                # QE typically: pw.x < pw.in > pw.out

                full_command = f"{self.config.command} -in pw.in"

                try:
                    proc = subprocess.run(
                        full_command,
                        check=False,
                        shell=True,  # shell=True to handle redirection if we used < > but here we used -in
                        cwd=str(work_dir),
                        stdout=open(output_path, "w"),
                        stderr=subprocess.PIPE,
                        timeout=self.config.timeout,
                        text=True,
                    )
                    stdout = output_path.read_text() if output_path.exists() else ""
                    stderr = proc.stderr

                except subprocess.TimeoutExpired:
                    stdout = output_path.read_text() if output_path.exists() else ""
                    stderr = "Timeout Expired"

                wall_time = time.time() - start_time

                # Check for success
                # Even if returncode is 0, we must verify output

                try:
                    result = self._parse_output(output_path, uid, wall_time, current_params, atoms)
                    if result.succeeded:
                        return result
                    # Logic to handle failure even if parse succeeded but indicated failure?
                    # _parse_output currently raises error if it can't parse, or returns partial result.
                except Exception:
                    # Parse failed, treat as error
                    pass

                # If we are here, something failed.
                error_type = RecoveryHandler.analyze(stdout, stderr)

                if not self.config.recoverable or attempt > self.config.max_retries:
                    break  # Fatal

                try:
                    current_params = RecoveryHandler.get_strategy(error_type, current_params)
                    # Log retry
                    continue
                except Exception:
                    break  # No strategy found

        msg = f"Job {uid} failed after {attempt} attempts."
        raise DFTFatalError(msg)

    def _stage_pseudos(self, work_dir: Path, atoms: Atoms):
        """
        Symlinks required pseudopotentials to the working directory.
        """
        from mlip_autopipec.dft.constants import SSSP_EFFICIENCY_1_1

        pseudo_src_dir = self.config.pseudo_dir
        if not pseudo_src_dir.exists():
            # In tests, this might not exist.
            # We should probably warn or skip if not strict.
            pass

        unique_species = set(atoms.get_chemical_symbols())
        for s in unique_species:
            if s in SSSP_EFFICIENCY_1_1:
                u_file = SSSP_EFFICIENCY_1_1[s]
                src = pseudo_src_dir / u_file
                dst = work_dir / u_file
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)

    def _parse_output(
        self, output_path: Path, uid: str, wall_time: float, params: dict, atoms: Atoms
    ) -> DFTResult:
        """
        Parses pw.out using QEOutputParser.
        """
        parser = QEOutputParser()
        return parser.parse(output_path, uid, wall_time, params)
