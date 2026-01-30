import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from mlip_autopipec.domain_models.calculation import (
    DFTConfig,
    DFTError,
    DFTResult,
)
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.input_gen import InputGenerator
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


class QERunner:
    """
    Runner for Quantum Espresso calculations with self-healing capabilities.
    """

    def __init__(self, config: DFTConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.recovery_handler = RecoveryHandler()

    def run(self, structure: Structure, job_id: str) -> DFTResult:
        """
        Run a DFT calculation for the given structure.
        """
        # Create job directory
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        current_params: dict[str, Any] = {}
        attempt = 1
        max_attempts = 10  # Safety break

        start_time = time.time()

        while attempt <= max_attempts:
            # Prepare input file content
            input_content = InputGenerator.generate_input(
                structure, self.config, parameters=current_params
            )

            input_path = job_dir / f"pw.in.{attempt}"
            output_path = job_dir / f"pw.out.{attempt}"

            input_path.write_text(input_content)

            # Execute
            try:
                with open(output_path, "w") as f_out:
                    # Construct command
                    # config.command might be "mpirun -np 4 pw.x"
                    # We split it for subprocess? No, use shell=True or split manually.
                    # Safer to assume command is shell-executable string.
                    # But providing input file usually requires: command < input_path
                    # or passing stdin.
                    # Let's use stdin redirection.

                    cmd = self.config.command.split()

                    # Also respect config.timeout
                    with open(input_path, "r") as f_in:
                        subprocess.run(
                            cmd,
                            stdin=f_in,
                            stdout=f_out,
                            stderr=subprocess.STDOUT,  # Capture stderr too
                            check=True,  # Raise CalledProcessError if fails
                            timeout=self.config.timeout,
                        )
            except subprocess.TimeoutExpired:
                # Handle timeout explicitly
                # This might leave output file incomplete.
                # RecoveryHandler handles WalltimeError.
                # We need to simulate a WalltimeError or let Parser detect it?
                # If process killed, output might contain "job killed" or nothing.
                # We should catch this and try recovery.
                # Let's see if we can parse the partial output.
                pass
            except subprocess.CalledProcessError:
                # QE failed (non-zero exit code).
                # Check output for specific errors.
                pass
            except FileNotFoundError:
                # Executable not found
                return DFTResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    work_dir=job_dir,
                    duration_seconds=time.time() - start_time,
                    log_content=f"Executable not found: {self.config.command}",
                    energy=0.0,
                    forces=np.zeros((len(structure.positions), 3)),
                )

            # Parse result
            try:
                # We parse the output file.
                # If subprocess failed, parsing might find the cause (e.g. MemoryError).
                result = DFTParser.parse(
                    output_path,
                    job_id=job_id,
                    work_dir=job_dir,
                    duration=time.time() - start_time,
                )
                return result

            except DFTError as e:
                # Recoverable error
                try:
                    current_params = self.recovery_handler.apply_fix(
                        current_params, e, attempt
                    )
                    attempt += 1
                    # Continue loop
                    continue
                except ValueError:
                    # Recovery failed (no more strategies)
                    # Or generic error
                    return DFTResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        work_dir=job_dir,
                        duration_seconds=time.time() - start_time,
                        log_content=f"Recovery failed after {attempt} attempts. Last error: {e}",
                        energy=0.0,
                        forces=np.zeros((len(structure.positions), 3)),
                    )

        # Max attempts reached
        return DFTResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            work_dir=job_dir,
            duration_seconds=time.time() - start_time,
            log_content="Max attempts reached without convergence.",
            energy=0.0,
            forces=np.zeros((len(structure.positions), 3)),
        )
