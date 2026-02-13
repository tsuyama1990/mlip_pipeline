from ase.data import atomic_numbers


class DeltaLearning:
    """Helper class for configuring Delta Learning baselines."""

    @staticmethod
    def get_config(elements: list[str], baseline: str | None) -> str:
        """
        Generates the reference potential configuration for Pacemaker.

        Args:
            elements: List of chemical elements (e.g., ["Fe", "Pt"]).
            baseline: The type of baseline potential (e.g., "zbl", "lj").

        Returns:
            A string containing the YAML configuration for the reference potential.
        """
        if not baseline:
            return ""

        baseline = baseline.lower()
        if baseline == "zbl":
            z_numbers = [str(atomic_numbers[el]) for el in elements]
            # Convert to comma-separated list string for YAML list
            z_list_str = ", ".join(z_numbers)
            return f"""
potential:
  delta:
    reference_potential:
      pair_style: zbl
      zbl_cut: 4.0
      zbl_z: [{z_list_str}]
"""
        if baseline == "lj":
            return """
potential:
  delta:
    reference_potential:
      pair_style: lj
      cutoff: 5.0
      epsilon: 1.0
      sigma: 1.0
"""
        msg = f"Unsupported delta learning baseline: {baseline}"
        raise ValueError(msg)
