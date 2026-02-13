from typing import Final

# Cycle 01 Constants
DEFAULT_WORK_DIR: Final[str] = "experiments"
DEFAULT_CONFIG_FILE: Final[str] = "config.yaml"

# Cycle 02 Constants
LAMMPS_MD_TEMPLATE: Final[str] = """
units metal
atom_style atomic
boundary p p p

# Potential setup (placeholder)
pair_style none

# MD Settings
velocity all create {temperature} 12345 dist gaussian
fix 1 all nvt temp {temperature} {temperature} 0.1
timestep 0.001

run {steps}
"""
