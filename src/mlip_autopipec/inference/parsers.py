"""
Log Parser Module.

This module provides functionality to parse LAMMPS log files for uncertainty data and halt status.
"""

from pathlib import Path


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

        max_gamma = 0.0
        halted = False
        halt_step = None

        content = log_file.read_text()

        # Check for halt message
        # "Fix halt condition met" is standard LAMMPS output for fix halt
        if "Fix halt condition met" in content:
            halted = True

        # Parse max_gamma from thermo output
        # Format: Step Temp c_max_gamma
        # We need to find the column index.
        lines = content.splitlines()
        header_found = False
        gamma_col_idx = -1
        step_col_idx = -1

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            if not header_found:
                if "Step" in parts and "c_max_gamma" in parts:
                    header_found = True
                    step_col_idx = parts.index("Step")
                    gamma_col_idx = parts.index("c_max_gamma")
            # Process data lines
            # Need to be careful about non-numeric lines (like "Loop time...")
            elif len(parts) > gamma_col_idx and len(parts) > step_col_idx:
                try:
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

        # If no explicit halt message but max_gamma is huge?
        # Rely on "Fix halt condition met" or specific error.

        if not halted:
            halt_step = None

        return max_gamma, halted, halt_step
