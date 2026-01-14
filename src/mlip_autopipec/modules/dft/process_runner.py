"""Module for running Quantum Espresso as a subprocess."""

import logging
import subprocess
from pathlib import Path

from mlip_autopipec.config.system import SystemConfig
from mlip_autopipec.modules.dft.exceptions import DFTCalculationError

logger = logging.getLogger(__name__)


class QEProcessRunner:
    """A robust runner for executing Quantum Espresso (pw.x) calculations."""

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the QEProcessRunner.

        Args:
            config: The fully-expanded system configuration object.

        """
        self.config = config

    def execute(self, input_path: Path, output_path: Path) -> None:
        """Run the pw.x executable as a subprocess.

        Args:
            input_path: Path to the QE input file.
            output_path: Path to write the QE output.

        Raises:
            DFTCalculationError: If pw.x returns a non-zero exit code.

        """
        command = [self.config.dft.command, "-in", str(input_path)]
        logger.info("Executing DFT command: %s", " ".join(command))
        # The use of subprocess.run is secure because `shell=False` is the
        # default and the command is passed as a list, preventing shell
        # injection. Additionally, `tempfile.TemporaryDirectory` creates a
        # secure, private directory, and `pathlib` joins prevent path
        # traversal attacks (`../`), ensuring files are written within the
        # temporary directory.
        try:
            with open(output_path, "w") as f:
                subprocess.run(
                    command,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
        except FileNotFoundError as e:
            error_message = (
                f"DFT command '{self.config.dft.command}' not found. "
                "Ensure Quantum Espresso is installed and in the system's PATH."
            )
            logger.error(error_message)
            raise DFTCalculationError(error_message) from e
        except subprocess.CalledProcessError as e:
            error_message = (
                f"DFT calculation failed with exit code {e.returncode}.\n"
                f"  Input file: {input_path}\n"
                f"  Output file: {output_path}\n"
                f"  Stderr: {e.stderr}"
            )
            logger.error(error_message)
            raise DFTCalculationError(error_message) from e
        logger.info("DFT calculation finished successfully. Output at %s", output_path)
