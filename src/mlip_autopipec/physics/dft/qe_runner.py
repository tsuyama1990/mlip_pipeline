import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, DFTError
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.job import JobStatus
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
            except subprocess.TimeoutExpired:
                 # Manually create a timeout error if subprocess raises it
                 # But we usually handle it inside _run_process or rely on checking output
                 # If subprocess.run raises TimeoutExpired, we catch it here.
                 pass
            except Exception as e:
                # Capture unexpected execution errors
                return DFTResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    work_dir=job_dir,
                    duration_seconds=time.time() - start_time,
                    log_content=f"Execution failed: {str(e)}",
                    energy=0.0, # Placeholder, maybe make optional? But strict schema requires float.
                    forces=[], # Placeholder
                    # Actually DFTResult fields are not Optional for energy/forces.
                    # We should probably return a Failed result differently or
                    # use a wrapping structure.
                    # But JobResult usually has status. If status is FAILED, maybe fields are ignored?
                    # Schema says `energy: float`, `forces: np.ndarray`. They are required.
                    # This is a schema design issue. Failed jobs usually don't have energy.
                    # I will assume for now we raise exception or return dummy values with FAILED status.
                    # Let's verify JobResult/DFTResult schema.
                    # DFTResult inherits JobResult.
                    # If I return DFTResult, I must provide energy.
                )
                # Wait, if I cannot return DFTResult because of missing fields, I should probably raise error
                # or the system design expects JobResult to support missing fields.
                # In Cycle 03 spec, DFTResult has required fields.
                # So `run` should probably raise DFTError if it fails completely,
                # or Orchestrator handles it.
                # I'll re-raise as DFTError if I can't return a valid Result.
                pass

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
                    # Construct a failed result if possible, or raise.
                    # Since we can't construct DFTResult without energy, we raise.
                    # Or we define a FailedDFTResult? No, strict schema.
                    # We raise the exception to the caller.
                    raise fatal

    def _run_process(self, config: DFTConfig, input_path: Path, output_path: Path):
        """
        Executes pw.x.
        """
        import shlex

        # Build command
        cmd_str = config.command
        if config.use_mpi:
            cmd_str = f"{config.mpi_command} {cmd_str}"

        # Add input flag
        cmd_str += f" -in {input_path.name}"

        # Split command
        cmd = shlex.split(cmd_str)

        with open(output_path, "w") as f_out:
            subprocess.run(
                cmd,
                cwd=input_path.parent,
                stdout=f_out,
                stderr=subprocess.STDOUT, # Capture stderr too
                timeout=config.timeout,
                check=False # We check output content for errors, not return code (QE often returns 0 even on failure)
            )
