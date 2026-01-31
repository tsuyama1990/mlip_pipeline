from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class LogParseResult:
    halt_detected: bool
    max_gamma: Optional[float] = None
    step: Optional[int] = None


class LammpsLogParser:
    """
    Parses LAMMPS logs to detect 'fix halt' conditions and extract metrics.
    """

    def parse(self, log_content: str) -> LogParseResult:
        halt_detected = False
        max_gamma = None
        step = None

        # Detect halt
        if "ERROR: Fix halt condition met" in log_content:
            halt_detected = True

        # Try to parse max gamma if available in thermo output
        # We look for a line with headers including 'c_pace_gamma' or 'v_max_gamma'
        # And then finding the max value in the data rows.

        lines = [line.strip() for line in log_content.splitlines()]
        header_indices: Dict[str, int] = {}

        for line in lines:
            if not line:
                continue

            if line.startswith("Step"):
                parts = line.split()
                header_indices = {} # Reset headers if new run starts
                for i, part in enumerate(parts):
                    header_indices[part] = i
            elif header_indices:
                # This is a data line (hopefully)
                parts = line.split()
                # Check if it looks like numbers
                try:
                    # Simple heuristic: first column is integer (Step)
                    if not parts:
                        continue

                    int(parts[0]) # Check if step is int

                    if len(parts) == len(header_indices):
                        # Extract Gamma
                        if "c_pace_gamma" in header_indices:
                             val = float(parts[header_indices["c_pace_gamma"]])
                             if max_gamma is None or val > max_gamma:
                                 max_gamma = val
                        elif "v_max_gamma" in header_indices:
                             val = float(parts[header_indices["v_max_gamma"]])
                             if max_gamma is None or val > max_gamma:
                                 max_gamma = val
                except ValueError:
                    pass # Not a data line

        return LogParseResult(halt_detected=halt_detected, max_gamma=max_gamma, step=step)
