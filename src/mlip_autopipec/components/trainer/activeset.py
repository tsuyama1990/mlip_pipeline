import logging
import os
import shutil
import subprocess
from pathlib import Path

from mlip_autopipec.utils.security import validate_safe_path

logger = logging.getLogger(__name__)


class ActiveSetSelector:
    """
    Selects a subset of structures using the MaxVol algorithm (via pace_activeset).
    """

    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit

    def select(self, input_path: Path, output_path: Path) -> Path:
        """
        Run pace_activeset to filter the dataset.

        Args:
            input_path: Path to the input dataset (gzipped pickle).
            output_path: Path to save the selected dataset.

        Returns:
            Path to the output dataset.
        """
        # Security: Validate paths are absolute and safe
        # We assume input_path is already validated by caller (Dataset export),
        # but we re-validate here for safety.
        # We also enforce that output_path is in the same directory structure or explicitly safe.

        try:
            safe_input = validate_safe_path(input_path)

            # Ensure output directory exists
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate output path is safe (e.g. not /etc/passwd)
            # validate_safe_path typically checks for traversal and absolute path
            # Since output file doesn't exist yet, we validate parent
            validate_safe_path(output_path.parent)
            safe_output = output_path.resolve()

        except Exception as e:
            msg = f"Path validation failed: {e}"
            raise ValueError(msg) from e

        if not safe_input.exists():
            msg = f"Input path does not exist: {safe_input}"
            raise FileNotFoundError(msg)

        # Configurable executable path
        pace_bin = os.environ.get("PACE_ACTIVESET_BIN", "pace_activeset")
        pace_executable = shutil.which(pace_bin)

        if not pace_executable:
            msg = f"pace_activeset executable '{pace_bin}' not found in PATH"
            raise RuntimeError(msg)

        # Construct command
        cmd = [
            pace_executable,
            "--data-filename",
            str(safe_input),
            "--output",
            str(safe_output),
            "--max",
            str(self.limit),
        ]

        logger.info(f"Running active set selection: {' '.join(cmd)}")

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
            logger.debug(f"pace_activeset output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            msg = f"pace_activeset failed with error: {e.stderr}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        if not output_path.exists():
            # If the command didn't create the file (mocking issue or tool failure not captured by exit code)
            # We log a warning.
            logger.warning(f"pace_activeset finished but {output_path} was not created.")

        return output_path
