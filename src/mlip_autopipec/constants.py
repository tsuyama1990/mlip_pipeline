import os

# File names
DEFAULT_CONFIG_FILENAME = os.getenv("MLIP_CONFIG_FILENAME", "config.yaml")

# Defaults
DEFAULT_PROJECT_NAME = os.getenv("MLIP_DEFAULT_PROJECT_NAME", "mlip_project")
DEFAULT_SEED = int(os.getenv("MLIP_DEFAULT_SEED", "42"))
DEFAULT_ELEMENTS = os.getenv("MLIP_DEFAULT_ELEMENTS", "Si").split(",")
DEFAULT_CUTOFF = float(os.getenv("MLIP_DEFAULT_CUTOFF", "5.0"))
DEFAULT_LOG_LEVEL = os.getenv("MLIP_DEFAULT_LOG_LEVEL", "INFO")

# Removed: DEFAULT_LOG_FILENAME, DEFAULT_STATE_FILENAME (Moved to Config)
