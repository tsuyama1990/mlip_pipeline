import logging
from pathlib import Path

log = logging.getLogger(__name__)


class AnalysisUtils:
    """
    Utilities for analyzing MD trajectories and logs.
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def get_properties(self) -> dict[str, float]:
        """
        Parses the LAMMPS log file to extract average properties.

        Returns:
            Dictionary of properties (temperature, pressure, potential_energy).
        """
        if not self.log_file.exists():
            return {}

        data = {"temperature": [], "pressure": [], "potential_energy": [], "total_energy": []}

        try:
            with open(self.log_file) as f:
                lines = f.readlines()

            in_thermo = False
            headers = []

            for line in lines:
                if "Step Temp" in line:
                    in_thermo = True
                    headers = line.split()
                    continue

                if "Loop time" in line:
                    in_thermo = False
                    continue

                if in_thermo:
                    try:
                        parts = line.split()
                        if len(parts) != len(headers):
                            continue

                        # Map columns
                        vals = {h: float(v) for h, v in zip(headers, parts)}

                        if "Temp" in vals:
                            data["temperature"].append(vals["Temp"])
                        if "Press" in vals:
                            data["pressure"].append(vals["Press"])
                        if "PotEng" in vals:
                            data["potential_energy"].append(vals["PotEng"])
                        if "TotEng" in vals:
                            data["total_energy"].append(vals["TotEng"])

                    except ValueError:
                        continue

            # Compute averages
            results = {}
            if data["temperature"]:
                results["temperature"] = sum(data["temperature"]) / len(data["temperature"])
            if data["pressure"]:
                results["pressure"] = sum(data["pressure"]) / len(data["pressure"])
            if data["potential_energy"]:
                results["potential_energy"] = sum(data["potential_energy"]) / len(
                    data["potential_energy"]
                )

            return results

        except Exception as e:
            log.error(f"Error parsing log file: {e}")
            return {}
