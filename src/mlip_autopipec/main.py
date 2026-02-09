import argparse
import sys
from pathlib import Path

from mlip_autopipec.constants import DEFAULT_CONFIG_PATH
from mlip_autopipec.core.orchestrator import Orchestrator


def main() -> None:
    """Entry point for PyAceMaker."""
    parser = argparse.ArgumentParser(description="PyAceMaker: MLIP Active Learning Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration YAML file.",
    )

    args = parser.parse_args()

    try:
        # Initialize Orchestrator
        # In Cycle 01, we just verify initialization works
        _ = Orchestrator(args.config)
        # Note: In future cycles, we will call orchestrator.run() here

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
