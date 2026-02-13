from ase.data import atomic_numbers

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.enums import HybridPotentialType


class HybridOverlay:
    """Helper class to generate LAMMPS hybrid/overlay pair commands."""

    def __init__(self, config: DynamicsConfig) -> None:
        self.config = config

    def get_pair_style(self) -> str:
        """
        Returns the pair_style command based on configuration.

        Returns:
            str: The LAMMPS pair_style command string.
        """
        base_style = "pair_style"
        potential_type = self.config.hybrid_potential

        if potential_type == HybridPotentialType.NONE or potential_type is None:
            return f"{base_style} pace"

        # Hybrid Overlay
        hybrid_args = ["hybrid/overlay", "pace"]

        if potential_type == HybridPotentialType.ZBL:
            inner = self.config.zbl_cut_inner
            outer = self.config.zbl_cut_outer
            hybrid_args.append(f"zbl {inner} {outer}")

        elif potential_type == HybridPotentialType.LJ:
            cut = self.config.lj_cutoff
            hybrid_args.append(f"lj/cut {cut}")

        return f"{base_style} {' '.join(hybrid_args)}"

    def get_pair_coeff(self, elements: list[str], potential_file: str) -> str:
        """
        Returns the pair_coeff commands.

        Args:
            elements: List of element symbols (e.g., ["Fe", "Pt"]).
            potential_file: Path/name of the potential file.

        Returns:
            str: The LAMMPS pair_coeff command string(s).
        """
        if not elements:
            msg = "Elements list cannot be empty for pair coefficients."
            raise ValueError(msg)

        commands = []

        # 1. PACE part (always present)
        # pair_coeff * * pace <file> <elements>
        elem_str = " ".join(elements)
        commands.append(f"pair_coeff * * pace {potential_file} {elem_str}")

        potential_type = self.config.hybrid_potential

        # 2. Baseline part
        if potential_type == HybridPotentialType.ZBL:
            # ZBL requires atomic numbers for each pair
            # pair_coeff i j zbl Z_i Z_j
            n_types = len(elements)
            for i in range(1, n_types + 1):
                for j in range(i, n_types + 1):
                    elem_i = elements[i - 1]
                    elem_j = elements[j - 1]

                    if elem_i not in atomic_numbers:
                        msg = f"Unknown element: {elem_i}"
                        raise ValueError(msg)
                    if elem_j not in atomic_numbers:
                        msg = f"Unknown element: {elem_j}"
                        raise ValueError(msg)

                    z_i = atomic_numbers[elem_i]
                    z_j = atomic_numbers[elem_j]
                    commands.append(f"pair_coeff {i} {j} zbl {z_i} {z_j}")

        elif potential_type == HybridPotentialType.LJ:
            # LJ
            epsilon = self.config.lj_epsilon
            sigma = self.config.lj_sigma
            commands.append(f"pair_coeff * * lj/cut {epsilon} {sigma}")

        return "\n".join(commands)
