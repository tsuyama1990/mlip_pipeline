import argparse
import sys
import yaml
from pathlib import Path

from mlip_autopipec.config.config_model import Config
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.logging_config import setup_logging

def main() -> None:
    parser = argparse.ArgumentParser(description="PYACEMAKER: Automated MLIP Construction")
    parser.add_argument("config", type=Path, help="Path to the config.yaml file.")

    args = parser.parse_args()
    config_path = args.config

    setup_logging()

    if not config_path.exists():
        sys.stderr.write(f"Error: Configuration file '{config_path}' does not exist.\n")
        sys.exit(1)

    try:
        with config_path.open("r") as f:
            data = yaml.safe_load(f)

        config = Config(**data)

        orchestrator = Orchestrator(config, work_dir=Path.cwd())
        orchestrator.run()

    except Exception as e:
        sys.stderr.write(f"Critical Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
