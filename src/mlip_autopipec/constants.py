import os

# File names
DEFAULT_CONFIG_FILENAME = os.getenv("MLIP_CONFIG_FILENAME", "config.yaml")

# Project Defaults
DEFAULT_PROJECT_NAME = os.getenv("MLIP_DEFAULT_PROJECT_NAME", "mlip_project")
DEFAULT_SEED = int(os.getenv("MLIP_DEFAULT_SEED", "42"))
DEFAULT_ELEMENTS = os.getenv("MLIP_DEFAULT_ELEMENTS", "Si").split(",")
DEFAULT_CUTOFF = float(os.getenv("MLIP_DEFAULT_CUTOFF", "5.0"))
DEFAULT_LOG_LEVEL = os.getenv("MLIP_DEFAULT_LOG_LEVEL", "INFO")

# Oracle Defaults (DFT/QE)
DEFAULT_K_POINTS_DENSITY = float(os.getenv("MLIP_DEFAULT_KPTS_DENSITY", "0.04"))  # 1/A
DEFAULT_SMEARING = float(os.getenv("MLIP_DEFAULT_SMEARING", "0.02"))  # Ry (approx 0.27 eV)
DEFAULT_SCF_K_POINTS = [2, 2, 2]  # Fallback if density not used

# Trainer Defaults (MACE/Pacemaker)
DEFAULT_TRAIN_EPOCHS = int(os.getenv("MLIP_DEFAULT_TRAIN_EPOCHS", "100"))
DEFAULT_MODEL_SIZE = os.getenv("MLIP_DEFAULT_MODEL_SIZE", "small") # small, medium, large

# Dynamics Defaults
DEFAULT_TIMESTEP = float(os.getenv("MLIP_DEFAULT_TIMESTEP", "1.0"))  # fs

# Adaptive Policy Defaults
DEFAULT_MD_MC_RATIO = float(os.getenv("MLIP_DEFAULT_MD_MC_RATIO", "0.1"))  # 10% MC steps
