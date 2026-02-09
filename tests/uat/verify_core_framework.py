import yaml
import sys
import shutil
from pathlib import Path
from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.domain_models.enums import GeneratorType, OracleType, TrainerType, DynamicsType, ValidatorType

# UAT Setup
UAT_DIR = Path("uat_workspace")
if UAT_DIR.exists():
    shutil.rmtree(UAT_DIR)
UAT_DIR.mkdir()

# Create a config file
config_data = {
    "orchestrator": {
        "work_dir": str(UAT_DIR),
        "max_cycles": 1,
        "uncertainty_threshold": 5.0,
    },
    "generator": {
        "type": str(GeneratorType.RANDOM),
        "seed": 42,
    },
    "oracle": {
        "type": str(OracleType.MOCK),
    },
    "trainer": {
        "type": str(TrainerType.MOCK),
    },
    "dynamics": {
        "type": str(DynamicsType.MOCK),
    },
    "validator": {
        "type": str(ValidatorType.MOCK),
    },
}

config_path = UAT_DIR / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config_data, f)

print(f"Created config at {config_path}")

# Run Orchestrator
try:
    # Mimic main.py loading
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Initialize logger
    setup_logging(UAT_DIR)

    orchestrator = Orchestrator(config)

    print("Orchestrator initialized successfully.")

    # Run cycle (minimal)
    orchestrator.run_cycle()
    print("Orchestrator cycle run completed.")

    # Verify directories
    if (UAT_DIR / "active_learning").exists() and (UAT_DIR / "potentials").exists():
        print("PASS: Directories created.")
    else:
        print("FAIL: Directories not created.")
        sys.exit(1)

    # Verify log file
    if (UAT_DIR / "orchestrator.log").exists():
        print("PASS: Log file created.")
    else:
        print("FAIL: Log file missing.")
        sys.exit(1)

except Exception as e:
    print(f"FAIL: Exception occurred: {e}")
    sys.exit(1)
