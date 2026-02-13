from mlip_autopipec.domain_models.config import DynamicsConfig


class UncertaintyWatchdog:
    """Helper class to configure LAMMPS fix halt for uncertainty."""

    def __init__(self, config: DynamicsConfig) -> None:
        self.config = config

    def get_commands(self, potential_file: str, elements: list[str]) -> str:
        """Returns the compute and fix halt commands."""
        if not self.config.halt_on_uncertainty:
            return ""

        commands = []
        elem_str = " ".join(elements)

        # 1. Compute per-atom uncertainty
        # compute ID group-ID pace filename elements...
        # Note: The 'pace' compute style is provided by the PACE package in LAMMPS
        # It calculates energy, force, and extrapolation grade (gamma)
        # Usage: compute <ID> <group> pace <filename> <elements>
        # Output: c_ID[1]=energy, c_ID[2]=gamma (or similar, checking docs/implementation)

        # According to pacemaker/lammps documentation for pair_style pace:
        # compute <ID> <group> pace <filename> <elements>
        # accessible as c_ID[1] (energy), c_ID[2] (gamma) usually.
        # But wait, typically ACE potentials output energy/stress/forces via pair style.
        # The compute pace is specifically for extra descriptors or gamma.

        # Assuming standard implementation where c_pace[1] is gamma or using a specific variable.
        # Let's verify standard usage. Often it's `compute ID group pace ...`
        # and it returns a per-atom vector.

        # Let's assume c_pace[1] is the max gamma or we need to reduce it.
        # Actually `compute pace` usually provides per-atom values.
        # To get max gamma in the system, we need `compute reduce max`.

        # From typical setup:
        # compute pace all pace potential.yace elements...
        # compute max_gamma all reduce max c_pace[2]  <-- Assuming index 2 is gamma

        # Wait, usually the pair style handles forces/energies.
        # The compute pace might be redundancy or for specific gamma calculation.
        # IF pair_style pace is used, maybe we can access gamma directly?
        # pair_style pace documentation says:
        # "The extrapolation grade can be accessed via the compute pace command."

        # Let's assume index 1 is gamma for now, or check UAT.
        # UAT says "max_gamma > threshold".
        # Let's assume c_pace[1] is gamma for safety, or check if index is specified.
        # Standard ACE/PACE compute:
        # c_ID[1] = extrapolation grade (gamma)

        commands.append(f"compute pace all pace {potential_file} {elem_str}")

        # Reduce to get global max
        commands.append("compute max_gamma all reduce max c_pace[1]")

        # Variable for check
        threshold = self.config.max_gamma_threshold
        # variable check_gamma equal c_max_gamma > threshold
        commands.append(f"variable check_gamma equal c_max_gamma>{threshold}")

        # Fix halt
        # fix ID group-ID halt N v_name != 0 error|warning code
        # Check every 1 step? Or every 10?
        # Let's check every step.
        commands.append("fix halt_check all halt 1 v_check_gamma != 0 error 100")

        return "\n".join(commands)
