import argparse
import sys
import logging
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
        # Security: Log detailed stack trace but show only generic/safe message to user
        if logging.getLogger().handlers:
            logging.critical(f"Fatal error: {e}", exc_info=True)
        print(f"Error running pipeline: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
