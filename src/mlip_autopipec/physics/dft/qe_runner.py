import subprocess
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, DFTError
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dft.input_gen import InputGenerator
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


class QERunner:
    def __init__(self, base_work_dir: Path = Path("_work_dft")):
        self.base_work_dir = base_work_dir
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.input_gen = InputGenerator()
        self.parser = DFTParser()
        self.recovery_handler = RecoveryHandler()
        self.max_retries = 3

    def _setup_pseudopotentials(self, work_dir: Path, config: DFTConfig):
        for el, src_path in config.pseudopotentials.items():
            if not src_path.exists():
                # For testing/mocking, we might not have real files.
                # Warn or error?
                # If strict, error.
                pass
            else:
                dst_path = work_dir / src_path.name
                if not dst_path.exists():
                    dst_path.symlink_to(src_path.resolve())

    def _execute(self, work_dir: Path, config: DFTConfig) -> tuple[str, str]:
        # Construct command: mpi_command + command + -in pw.in
        # Example: "mpirun -np 4" + "pw.x" + "-in pw.in"

        cmd_str = f"{config.mpi_command} {config.command} -in pw.in"
        cmd_list = cmd_str.split()

        # Simple check for executable (optional, can just let subprocess fail)

        try:
            result = subprocess.run(
                cmd_list,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=config.timeout,
                check=False,  # Allow non-zero exit for parsing errors
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            # Re-raise as our specific error or handle here?
            # It's better to catch it in the loop
            raise

    def run(self, structure: Structure, config: DFTConfig) -> DFTResult:
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.base_work_dir / f"job_{timestamp}_{job_id[:8]}"
        work_dir.mkdir()

        current_config = config
        # Initial attempt

        start_time = datetime.now()

        # Setup pseudos
        # We wrap in try to catch setup errors
        try:
            self._setup_pseudopotentials(work_dir, config)
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return self._failed_result(
                job_id, work_dir, duration, f"Setup failed: {e}", structure
            )

        final_stdout = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                # 1. Generate Input
                input_content = self.input_gen.generate_input(structure, current_config)
                (work_dir / "pw.in").write_text(input_content)

                # 2. Execute
                try:
                    stdout, stderr = self._execute(work_dir, current_config)
                    final_stdout = stdout
                except subprocess.TimeoutExpired:
                    raise DFTError("Timeout Expired")

                # 3. Parse
                parsed_data = self.parser.parse_output(stdout, stderr)

                duration = (datetime.now() - start_time).total_seconds()

                return DFTResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    work_dir=work_dir,
                    duration_seconds=duration,
                    log_content=stdout[-2000:],  # Keep tail
                    energy=parsed_data["energy"],
                    forces=parsed_data["forces"],
                    stress=parsed_data["stress"],
                    magmoms=None,
                )

            except DFTError as e:
                # If we are at max retries, fail
                if attempt == self.max_retries:
                    duration = (datetime.now() - start_time).total_seconds()
                    return self._failed_result(
                        job_id,
                        work_dir,
                        duration,
                        f"Failed after {attempt} attempts. Last error: {str(e)}\nLog tail: {final_stdout[-1000:]}",
                        structure,
                    )

                # Recovery
                try:
                    current_config = self.recovery_handler.apply_fix(
                        current_config, e, attempt
                    )
                except DFTError:
                    # Recovery failed (no fix known)
                    duration = (datetime.now() - start_time).total_seconds()
                    return self._failed_result(
                        job_id,
                        work_dir,
                        duration,
                        f"Unrecoverable error: {str(e)}",
                        structure,
                    )

        # Fallback (should be unreachable)
        return self._failed_result(job_id, work_dir, 0.0, "Unknown failure", structure)

    def _failed_result(self, job_id, work_dir, duration, log, structure):
        """Helper to return a FAILED result with dummy data."""
        return DFTResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            work_dir=work_dir,
            duration_seconds=duration,
            log_content=log,
            energy=0.0,
            forces=np.zeros((len(structure.positions), 3)),
            stress=None,
        )
