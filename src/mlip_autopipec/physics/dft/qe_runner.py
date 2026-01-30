import subprocess
import time
import uuid
import shlex
from pathlib import Path
from typing import Optional

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, DFTError
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.input_gen import InputGenerator
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


class QERunner:
    """
    Orchestrates running Quantum Espresso calculations with automatic recovery.
    """

    def __init__(self, work_dir: Path, recovery_handler: Optional[RecoveryHandler] = None):
        self.work_dir = work_dir
        self.recovery_handler = recovery_handler or RecoveryHandler()

    def run(self, structure: Structure, config: DFTConfig, job_id: Optional[str] = None) -> DFTResult:
        if job_id is None:
            job_id = f"dft_{uuid.uuid4().hex[:8]}"

        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        current_config = config.model_copy()
        parser = DFTParser()

        attempt = 0

        while True:
            attempt += 1

            # Prepare file paths
            input_file = job_dir / f"pw.in.{attempt}"
            output_file = job_dir / f"pw.out.{attempt}"

            # Generate Input
            igen = InputGenerator(current_config)
            input_content = igen.generate(structure)
            input_file.write_text(input_content)

            # Run
            start_time = time.time()
            try:
                self._run_process(current_config, input_file, output_file)
            except Exception as e:
                # Capture unexpected execution errors and raise DFTError
                # We do NOT return a malformed DFTResult.
                raise DFTError(f"Execution failed for job {job_id}: {str(e)}") from e

            duration = time.time() - start_time

            # Read output
            if output_file.exists():
                output_content = output_file.read_text()
            else:
                output_content = ""

            try:
                # Parse
                result = parser.parse(output_content, job_id=job_id, work_dir=job_dir)
                result.duration_seconds = duration # Update duration
                return result

            except DFTError as e:
                # Recovery
                print(f"Attempt {attempt} failed: {e}. Trying recovery...")
                try:
                    current_config = self.recovery_handler.apply_fix(current_config, e, attempt)
                except RuntimeError as fatal:
                    print(f"Recovery failed: {fatal}")
                    raise fatal

    def _run_process(self, config: DFTConfig, input_path: Path, output_path: Path):
        """
        Executes pw.x.
        """
        # Validate mpi_command to prevent basic injection if it's dynamic
        if config.use_mpi:
            if ";" in config.mpi_command or "|" in config.mpi_command:
                raise ValueError("Invalid characters in mpi_command")

        # Build command safely
        # We assume config.command and config.mpi_command are trustworthy configuration values
        # but strictly splitting them ensures we don't execute arbitrary shell strings if we avoid shell=True.

        cmd_parts = []
        if config.use_mpi:
             cmd_parts.extend(shlex.split(config.mpi_command))

        cmd_parts.extend(shlex.split(config.command))

        # Add input flag
        # QE typically uses -in or < input. -in is safer for command line arg.
        cmd_parts.append("-in")
        cmd_parts.append(input_path.name)

        with open(output_path, "w") as f_out:
            subprocess.run(
                cmd_parts,
                cwd=input_path.parent,
                stdout=f_out,
                stderr=subprocess.STDOUT, # Capture stderr too
                timeout=config.timeout,
                check=False # We check output content for errors, not return code
            )
