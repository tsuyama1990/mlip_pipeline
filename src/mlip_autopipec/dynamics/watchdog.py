import logging

from mlip_autopipec.domain_models.config import DynamicsConfig

logger = logging.getLogger(__name__)


class UncertaintyWatchdog:
    """Helper class to configure LAMMPS fix halt for uncertainty."""

    def __init__(self, config: DynamicsConfig) -> None:
        self.config = config

    def get_commands(self, potential_file: str, elements: list[str]) -> str:
        """
        Returns the compute and fix halt commands.

        Args:
            potential_file: Path to the potential file.
            elements: List of element symbols.

        Returns:
            str: The LAMMPS commands to enable uncertainty watchdog.
        """
        if not potential_file:
            msg = "Potential file path cannot be empty."
            raise ValueError(msg)

        if not self.config.halt_on_uncertainty:
            return ""

        commands = []
        elem_str = " ".join(elements)

        # 1. Compute per-atom uncertainty
        # Use specific ID 'pace_gamma' to avoid conflict
        # gamma_mode=1 enables extrapolation grade calculation
        commands.append(f"compute pace_gamma all pace {potential_file} {elem_str} gamma_mode=1")

        # Reduce to get global max
        # Assuming pace_gamma produces a scalar per atom representing gamma
        commands.append("compute max_gamma all reduce max c_pace_gamma")

        # Variable for check
        threshold = self.config.max_gamma_threshold
        commands.append(f"variable check_gamma equal c_max_gamma>{threshold}")

        # Fix halt
        # fix ID group-ID halt N v_name != 0 error|warning code
        commands.append("fix halt_check all halt 1 v_check_gamma != 0 error 100")

        return "\n".join(commands)
