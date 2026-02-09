import argparse
import sys
from pathlib import Path
from mlip_autopipec.core.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="PyAceMaker: Automated MLIP Pipeline")
    parser.add_argument("config", help="Path to the configuration file (YAML)")

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        orchestrator = Orchestrator(config_path)
        orchestrator.run_cycle()
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
