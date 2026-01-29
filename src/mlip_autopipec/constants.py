"""Global constants for the MLIP AutoPipec project."""


# File names
DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_STATE_FILENAME = "workflow_state.json"
DEFAULT_LOG_FILENAME = "mlip_pipeline.log"

# Logging
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Directory structure
WORK_DIR_NAME = "_work"
DATA_DIR_NAME = "data"
POTENTIALS_DIR_NAME = "potentials"

# Defaults
DEFAULT_PROJECT_NAME = "mlip_project"
DEFAULT_SEED = 42
DEFAULT_ELEMENTS = ["Si"]
DEFAULT_CUTOFF = 5.0
DEFAULT_LOG_LEVEL = "INFO"
