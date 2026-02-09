from pathlib import Path

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "mlip_autopipec"

# Default Paths
DEFAULT_WORK_DIR = Path("work")
DEFAULT_CONFIG_PATH = Path("config.yaml")

# Logic Constants
MAX_CYCLES = 10
DEFAULT_SEED = 42

# Component Defaults
DEFAULT_POTENTIAL_FORMAT = "yace"
