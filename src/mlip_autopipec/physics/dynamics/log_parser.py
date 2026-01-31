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
    Supports incremental parsing to avoid loading large logs into memory.
    """

    def __init__(self):
        self.halt_detected = False
        self.max_gamma: Optional[float] = None
        self.header_indices: Dict[str, int] = {}

    def parse_line(self, line: str) -> None:
        """
        Parse a single line of the log file and update state.
        """
        line = line.strip()
        if not line:
            return

        if "ERROR: Fix halt condition met" in line:
            self.halt_detected = True

        # Thermo Data Parsing
        if line.startswith("Step"):
            parts = line.split()
            self.header_indices = {} # Reset headers if new run starts
            for i, part in enumerate(parts):
                self.header_indices[part] = i
        elif self.header_indices:
            # This is a data line (hopefully)
            parts = line.split()
            try:
                # Simple heuristic: first column is integer (Step)
                if not parts:
                    return

                int(parts[0]) # Check if step is int

                if len(parts) == len(self.header_indices):
                    # Extract Gamma
                    current_gamma = None
                    if "c_pace_gamma" in self.header_indices:
                         current_gamma = float(parts[self.header_indices["c_pace_gamma"]])
                    elif "v_max_gamma" in self.header_indices:
                         current_gamma = float(parts[self.header_indices["v_max_gamma"]])

                    if current_gamma is not None:
                        if self.max_gamma is None or current_gamma > self.max_gamma:
                            self.max_gamma = current_gamma

            except ValueError:
                pass # Not a data line

    def get_result(self) -> LogParseResult:
        return LogParseResult(
            halt_detected=self.halt_detected,
            max_gamma=self.max_gamma
        )

    def parse(self, log_content: str) -> LogParseResult:
        """Legacy method for backward compatibility/testing with full string."""
        # Reset state for fresh parse
        self.halt_detected = False
        self.max_gamma = None
        self.header_indices = {}

        for line in log_content.splitlines():
            self.parse_line(line)

        return self.get_result()
