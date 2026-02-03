import logging
from pathlib import Path

from mlip_autopipec.domain_models.dynamics import MDResult, MDStatus

logger = logging.getLogger(__name__)


class LogParser:
    def parse(self, log_path: Path) -> MDResult:
        if not log_path.exists():
            logger.error(f"Log file not found: {log_path}")
            return MDResult(status=MDStatus.FAILED)

        content = log_path.read_text()

        # Check for Halt
        if "ERROR: Fix halt condition met" in content:
            logger.info("Detected HALT in LAMMPS log")
            halt_step = self._extract_last_step(content)
            return MDResult(status=MDStatus.HALTED, log_path=log_path, halt_step=halt_step)

        # Check for Success
        if "Loop time of" in content:
            return MDResult(status=MDStatus.COMPLETED, log_path=log_path)

        # Default to Failed
        logger.warning("LAMMPS log indicates failure or incomplete run")
        return MDResult(status=MDStatus.FAILED, log_path=log_path)

    def _extract_last_step(self, content: str) -> int | None:
        """
        Extracts the last timestep recorded in the log.
        Assumes standard thermo output format where the first column is the step.
        """
        lines = content.splitlines()
        last_step = None

        # Simple heuristic: Look for lines starting with an integer,
        # that have at least 3 columns (Step, Temp, PotEng, etc.)
        # and where subsequent columns are also numbers.

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0].isdigit():
                try:
                    # Check if at least 3 columns
                    if len(parts) < 3:
                        continue

                    # Check if 2nd column is a number (float or int)
                    try:
                        float(parts[1])
                    except ValueError:
                        continue

                    step = int(parts[0])
                    last_step = step
                except ValueError:
                    pass

        return last_step
