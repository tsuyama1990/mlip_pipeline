import os

DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_LOG_FILENAME = "mlip_pipeline.log"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_PROJECT_NAME = "MyMLIPProject"
# Elements can be overridden by env var MLIP_DEFAULT_ELEMENTS (comma-separated)
DEFAULT_ELEMENTS = os.environ.get("MLIP_DEFAULT_ELEMENTS", "Si").split(",")
DEFAULT_CUTOFF = 5.0
DEFAULT_SEED = 42
