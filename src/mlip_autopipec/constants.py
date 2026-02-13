
# --- DEFAULT CONSTANTS ---

# Default LAMMPS template
DEFAULT_LAMMPS_TEMPLATE = """
# Default LAMMPS input script
# This is a placeholder. Please configure 'lammps_template' in GeneratorConfig.
"""

# Default Simulation Parameters
DEFAULT_MD_STEPS = 1000
DEFAULT_TIMESTEP = 0.001
DEFAULT_TEMPERATURE = 300.0

# Default DFT Parameters
DEFAULT_KSPACING = 0.04
DEFAULT_ENCUT = 60.0  # Ry
DEFAULT_MIXING_BETA = 0.7
DEFAULT_SMEARING_WIDTH = 0.01  # Ry

# Default Trainer Parameters
DEFAULT_MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_CUTOFF = 5.0
DEFAULT_ORDER = 2
DEFAULT_BASIS_SIZE = 500

# Default Orchestrator Parameters
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_MAX_CYCLES = 1
