"""
Analysis Utilities Module.

This module provides utility classes for analyzing simulation outputs,
such as parsing thermodynamic data from log files.
"""

import statistics
from pathlib import Path


class AnalysisUtils:
    """Utilities for analyzing LAMMPS log files."""

    def __init__(self, log_file: Path) -> None:
        """
        Initialize the AnalysisUtils with a log file path.

        Args:
            log_file: Path to the LAMMPS log file.
        """
        self.log_file = log_file

    def get_thermo_stats(self) -> dict[str, float]:
        """
        Parses LAMMPS log file to extract thermodynamic statistics.

        Returns:
            A dictionary containing mean temperature, pressure, etc.

        Raises:
            FileNotFoundError: If the log file does not exist.
        """
        if not self.log_file.exists():
            msg = f"Log file not found: {self.log_file}"
            raise FileNotFoundError(msg)

        content = self.log_file.read_text()

        # Simple parsing logic
        # Look for "Step Temp Press ..." line
        # Collect data lines until "Loop time ..."

        data = []
        headers = []
        parsing = False

        for line in content.splitlines():
            parts = line.split()
            if not parts:
                continue

            if "Step" in parts and "Temp" in parts:
                headers = parts
                parsing = True
                continue

            if parsing:
                if "Loop" in parts and "time" in parts:
                    parsing = False
                    break

                # Try to parse numbers
                try:
                    values = [float(x) for x in parts]
                    if len(values) == len(headers):
                        row = dict(zip(headers, values, strict=False))
                        data.append(row)
                except ValueError:
                    continue

        if not data:
            return {}

        stats = {}
        # Calculate means
        if "Temp" in headers:
            temps = [d["Temp"] for d in data]
            stats["temperature_mean"] = statistics.mean(temps)
            stats["temperature_std"] = (
                statistics.stdev(temps) if len(temps) > 1 else 0.0
            )

        if "Press" in headers:
            presss = [d["Press"] for d in data]
            stats["pressure_mean"] = statistics.mean(presss)

        stats["steps"] = len(data)

        return stats
