from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig


class ScriptGenerator:
    """
    Generates LAMMPS input scripts for MD simulations with ACE potentials and Uncertainty Quantification.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

    def generate(self, atoms: Atoms, working_dir: Path, structure_file: Path) -> str:
        """
        Generates the content of the LAMMPS input script.

        Args:
            atoms: The ASE Atoms object (used for symbol extraction).
            working_dir: The directory where the simulation will run.
            structure_file: The path to the LAMMPS data file (relative to working_dir).

        Returns:
            The content of the input script.
        """
        symbols = sorted(set(atoms.get_chemical_symbols()))
        potential_path = self.config.potential_path

        # Use local filename if potential_path is provided (since Runner copies it), otherwise default
        potential_filename = potential_path.name if potential_path else "pot.yace"

        # Determine fix command based on ensemble
        if self.config.ensemble == "npt":
            # standard NPT damping parameters: Tdamp=100*dt, Pdamp=1000*dt usually
            # Here using hardcoded damping 0.1 and 1.0 ps as in typical examples or Spec
            fix_cmd = (
                f"fix             1 all npt temp {self.config.temperature} {self.config.temperature} 0.1 "
                f"iso {self.config.pressure} {self.config.pressure} 1.0"
            )
        else:
            fix_cmd = f"fix             1 all nvt temp {self.config.temperature} {self.config.temperature} 0.1"

        script = f"""
units           metal
boundary        p p p
atom_style      atomic

read_data       {structure_file.name}

pair_style      pace
pair_coeff      * * {potential_filename} {" ".join(symbols)}

thermo          {self.config.sampling_interval}
thermo_style    custom step temp pe etotal press

# Uncertainty Quantification
compute         pace all pace
compute         max_gamma all reduce max c_pace[1]

# Dump high uncertainty frames
dump            uncert_dump all custom {self.config.sampling_interval} dump.gamma id type x y z c_pace[1]
dump_modify     uncert_dump thresh c_max_gamma > {self.config.uq_threshold}

timestep        {self.config.timestep}
{fix_cmd}

run             {self.config.steps}
"""
        return script
