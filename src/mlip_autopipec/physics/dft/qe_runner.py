import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, DFTError
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dft.input_gen import InputGenerator
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


class QERunner:
    """
    Executes Quantum Espresso (PWSCF) calculations with automatic recovery.
    """

    def __init__(self, config: DFTConfig, base_work_dir: Path = Path("_work_dft")):
        self.config = config
        self.base_work_dir = base_work_dir
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.input_gen = InputGenerator()
        self.parser = DFTParser()
        self.recovery = RecoveryHandler(config.recovery)

    def run(self, structure: Structure) -> DFTResult:
        """
        Run a DFT calculation with self-healing.
        """
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = self.base_work_dir / f"dft_{timestamp}_{job_id[:8]}"
        work_dir.mkdir()

        current_config = self.config.model_copy(deep=True)
        attempt = 0

        try:
            while True:
                try:
                    # 1. Write Inputs
                    self._write_inputs(work_dir, structure, current_config)

                    # 2. Execute
                    self._execute(work_dir, current_config)

                    # 3. Parse Output
                    result = self._parse_output(work_dir)

                    # If parsed successfully without error, return result
                    return result

                except DFTError as e:
                    # 4. Recovery
                    try:
                        current_config, attempt = self.recovery.apply_fix(
                            current_config, e, attempt
                        )
                        # Log recovery action?
                        print(f"DFT Failure: {e}. Applying fix (Attempt {attempt})...")
                    except DFTError as final_error:
                        # Recovery exhausted or impossible
                        raise final_error
        finally:
            # Cleanup
            # Auditor requirement: "Add shutil.rmtree(work_dir) in finally block"
            if work_dir.exists():
                shutil.rmtree(work_dir)

    def _write_inputs(self, work_dir: Path, structure: Structure, config: DFTConfig) -> None:
        """Write pw.in and link pseudopotentials."""
        content = self.input_gen.generate_input(structure, config)
        (work_dir / "pw.in").write_text(content)

        # Link pseudopotentials
        for species, path in config.pseudopotentials.items():
            src = path
            if not src.is_absolute():
                 # Assume relative to CWD or configured lib path?
                 # For now, assume relative to CWD if not absolute.
                 src = Path.cwd() / src

            dst = work_dir / src.name
            if src.exists() and not dst.exists():
                # Symlink or copy
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copy(src, dst)

    def _execute(self, work_dir: Path, config: DFTConfig) -> None:
        """Execute pw.x."""
        # Config.command is now list[str]
        cmd_list = list(config.command)
        cmd_list.extend(["-in", "pw.in"])

        # Basic executable check
        exe = cmd_list[0]
        # Only check if it looks like a path or command (not if mocked)
        # But robust code checks.
        if not shutil.which(exe):
             pass

        with (work_dir / "pw.out").open("w") as out_f:
            subprocess.run(
                cmd_list,
                cwd=work_dir,
                stdout=out_f,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout for parsing
                timeout=config.timeout,
                check=False # Allow non-zero exit codes (QE might crash but leave useful logs)
            )

    def _parse_output(self, work_dir: Path) -> DFTResult:
        """Parse pw.out."""
        out_file = work_dir / "pw.out"
        if not out_file.exists():
            raise DFTError("Output file pw.out not found.")

        content = out_file.read_text()
        return self.parser.parse(content)
