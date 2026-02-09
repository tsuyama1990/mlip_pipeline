# Constants for MLIP Pipeline

# Physics Constants
MAX_ATOMIC_NUMBER = 118
MAX_FORCE_MAGNITUDE = 1000.0  # eV/A
MAX_ENERGY_MAGNITUDE = 1e6  # eV (per structure total)
MAX_VACUUM_SIZE = 50.0  # Angstroms

# Oracle / DFT Defaults
HEALER_MIXING_BETA_TARGET = 0.3
HEALER_DEGAUSS_TARGET = 0.02

# Orchestrator Defaults
DEFAULT_BUFFER_SIZE = 1000

# LAMMPS Templates
LAMMPS_TEMPLATE_UNITS = "units           metal"
LAMMPS_TEMPLATE_ATOM_STYLE = "atom_style      atomic"
LAMMPS_TEMPLATE_BOUNDARY = "boundary        p p p"
LAMMPS_TEMPLATE_VELOCITY = "velocity        all create {temperature} 12345 dist gaussian"
LAMMPS_TEMPLATE_FIX_NVE = "fix             1 all nve"
LAMMPS_TEMPLATE_DUMP = "dump            1 all custom {thermo_freq} {dump_filename} id type x y z fx fy fz"
