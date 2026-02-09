MAX_VACUUM_SIZE: float = 50.0
HEALER_MIXING_BETA_TARGET: float = 0.3
HEALER_DEGAUSS_TARGET: float = 0.02
DEFAULT_BUFFER_SIZE: int = 1000
EON_DEFAULT_TIME_STEP: float = 1.0

# LAMMPS Templates (Placeholders for now, will be filled in Cycle 05)
LAMMPS_TEMPLATE_HEADER = """
units metal
boundary p p p
atom_style atomic
"""

LAMMPS_TEMPLATE_MINIMIZE = """
minimize 1.0e-4 1.0e-6 100 1000
"""

LAMMPS_TEMPLATE_MD = """
velocity all create ${temp} 12345
fix 1 all nvt temp ${temp} ${temp} 0.1
run ${steps}
"""
