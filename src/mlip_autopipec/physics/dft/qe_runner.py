import shutil
import subprocess
from pathlib import Path

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTError, DFTResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.input_gen import InputGenerator
from mlip_autopipec.physics.dft.parser import Parser
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


class QERunner:
    """Executes Quantum Espresso calculations with automatic recovery."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_handler = RecoveryHandler()

    def run(self, structure: Structure, config: DFTConfig) -> DFTResult:
        """
        Run the DFT calculation. Retries on failure using RecoveryHandler.
        """
        current_config = config

        # Loop: attempt 0 to MAX_ATTEMPTS - 1
        # MAX_ATTEMPTS is the total number of allowed executions.
        # e.g., if MAX_ATTEMPTS=5, we try 0, 1, 2, 3, 4.

        for attempt in range(self.recovery_handler.MAX_ATTEMPTS):
            try:
                # pass attempt + 1 to execution/recovery if they expect 1-based index (logging/logic)
                return self._execute_single_run(structure, current_config, attempt + 1)
            except DFTError as e:
                # If we are at the last attempt (MAX_ATTEMPTS - 1), we can't recover
                if attempt == self.recovery_handler.MAX_ATTEMPTS - 1:
                    raise e

                # Try to recover
                # We catch the error, log it (implicitly via flow), and ask for a fix
                try:
                    # Pass attempt+1 because RecoveryHandler logic uses 1-based indexing for strategies
                    current_config = self.recovery_handler.apply_fix(current_config, e, attempt + 1)
                except DFTError as fatal_e:
                    # If recovery fails (e.g. unknown error), raise original or fatal
                    raise fatal_e

        raise DFTError("Unexpected termination of retry loop")

    def _execute_single_run(self, structure: Structure, config: DFTConfig, attempt: int) -> DFTResult:
        run_dir = self.working_dir / f"run_{attempt}"
        run_dir.mkdir(exist_ok=True)

        # 1. Prepare Environment (Pseudopotentials)
        for _, p_path in config.pseudopotentials.items():
            target = run_dir / p_path.name
            if not target.exists():
                # Resolve source path
                src = p_path.resolve()
                if not src.exists():
                     # Should have been validated earlier, but good check
                     pass # Let it fail later or warn?

                try:
                    target.symlink_to(src)
                except OSError:
                    # Fallback to copy
                    shutil.copy(src, target)

        # 2. Generate Input
        input_content = InputGenerator.generate(structure, config)
        input_path = run_dir / "pw.in"
        input_path.write_text(input_content)

        output_path = run_dir / "pw.out"

        # 3. Execute
        cmd = config.command.split()

        with open(output_path, "w") as f_out, open(input_path, "r") as f_in:
            try:
                subprocess.run(
                    cmd,
                    stdin=f_in,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,
                    cwd=run_dir,
                    check=True,
                    timeout=config.timeout
                )
            except subprocess.TimeoutExpired:
                 raise DFTError(f"Calculation timed out after {config.timeout}s")
            except subprocess.CalledProcessError:
                # Process failed (non-zero exit code).
                # QE often exits with error code on crash.
                # We proceed to parsing, as Parser handles error messages in output.
                # If output is empty or parser fails, we raise DFTError.
                pass
            except FileNotFoundError:
                 raise DFTError(f"Command not found: {cmd[0]}")

        # 4. Parse Output
        return Parser.parse(output_path)
