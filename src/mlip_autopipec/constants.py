"""Global constants for the MLIP AutoPipec project."""

import ast
import os


def _get_env_list(key: str, default: list[str]) -> list[str]:
    val = os.getenv(key)
    if val:
        try:
            # Safe eval for list strings like "['Si', 'C']"
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            # Fallback to splitting by comma if not a list literal
            return [x.strip() for x in val.split(",")]
    return default

def _get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default

def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default

# File names
DEFAULT_CONFIG_FILENAME = os.getenv("MLIP_CONFIG_FILENAME", "config.yaml")
DEFAULT_STATE_FILENAME = os.getenv("MLIP_STATE_FILENAME", "workflow_state.json")
DEFAULT_LOG_FILENAME = os.getenv("MLIP_LOG_FILENAME", "mlip_pipeline.log")

# Logging
LOG_FORMAT = os.getenv("MLIP_LOG_FORMAT", "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
DATE_FORMAT = os.getenv("MLIP_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

# Directory structure
WORK_DIR_NAME = os.getenv("MLIP_WORK_DIR", "_work")
DATA_DIR_NAME = os.getenv("MLIP_DATA_DIR", "data")
POTENTIALS_DIR_NAME = os.getenv("MLIP_POTENTIALS_DIR", "potentials")

# Defaults
DEFAULT_PROJECT_NAME = os.getenv("MLIP_DEFAULT_PROJECT_NAME", "mlip_project")
DEFAULT_SEED = _get_env_int("MLIP_DEFAULT_SEED", 42)
DEFAULT_ELEMENTS = _get_env_list("MLIP_DEFAULT_ELEMENTS", ["Si"])
DEFAULT_CUTOFF = _get_env_float("MLIP_DEFAULT_CUTOFF", 5.0)
DEFAULT_LOG_LEVEL = os.getenv("MLIP_DEFAULT_LOG_LEVEL", "INFO")
