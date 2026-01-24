"""
Log Parser Module.

This module provides functionality to parse LAMMPS log files for uncertainty data and halt status.
"""

from pathlib import Path
import os

# Maximum log file size to process (100 MB) to prevent OOM
MAX_LOG_SIZE = 100 * 1024 * 1024

class LogParser:
    """Parses LAMMPS log files."""

    @staticmethod
    def parse(log_file: Path) -> tuple[float, bool, int | None]:
        """
        Parses the log file to extract max gamma and halt status.

        Args:
            log_file: Path to the LAMMPS log file.

        Returns:
            Tuple of (max_gamma, halted, halt_step).
            max_gamma: The maximum c_max_gamma observed.
            halted: True if the simulation was halted due to uncertainty.
            halt_step: The step at which it halted (or None).
        """
        if not log_file.exists():
            return 0.0, False, None

        # Scalability Check: Ensure file is not too large
        if log_file.stat().st_size > MAX_LOG_SIZE:
             # If too large, we might skip parsing or parse only tail.
             # For now, we return default to avoid crashing.
             # Alternatively, read line by line.
             # Let's switch to line-by-line reading for memory efficiency.
             pass

        max_gamma = 0.0
        halted = False
        halt_step = None

        header_found = False
        gamma_col_idx = -1
        step_col_idx = -1

        try:
             with log_file.open("r", encoding="utf-8", errors="replace") as f:
                 for line in f:
                    # Check for halt message
                    if "Fix halt condition met" in line:
                        halted = True

                    parts = line.split()
                    if not parts:
                        continue

                    if not header_found:
                        if "Step" in parts and "c_max_gamma" in parts:
                            header_found = True
                            step_col_idx = parts.index("Step")
                            gamma_col_idx = parts.index("c_max_gamma")
                    # Process data lines
                    elif len(parts) > gamma_col_idx and len(parts) > step_col_idx:
                        try:
                            # Verify if parts look like numbers
                            step_val = int(parts[step_col_idx])
                            gamma_val = float(parts[gamma_col_idx])
                            max_gamma = max(max_gamma, gamma_val)

                            # If halted, the last valid step might be the halt step
                            # But LAMMPS prints the error AFTER the step line usually.
                            # Or the step line IS the last line.
                            # We update halt_step blindly; the last one seen is the last one run.
                            halt_step = step_val
                        except ValueError:
                            pass
        except Exception:
             # Handle read errors gracefully
             return max_gamma, halted, halt_step

        if not halted:
            halt_step = None

        return max_gamma, halted, halt_step
