from pathlib import Path

# Operational Constants
MAX_VACUUM_SIZE = 50.0
HEALER_MIXING_BETA_TARGET = 0.3
HEALER_DEGAUSS_TARGET = 0.02
DEFAULT_BUFFER_SIZE = 1000
EON_DEFAULT_TIME_STEP = 1.0

# Paths
DEFAULT_WORK_DIR = Path("work_dir")
DEFAULT_CONFIG_FILE = Path("config.yaml")

# LAMMPS Templates
LAMMPS_TEMPLATE_STATIC = """
# Placeholder for static calculation
"""
