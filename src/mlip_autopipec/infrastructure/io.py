from pathlib import Path
from typing import Any
import subprocess
import logging

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Safely load a YAML file.

    WARNING: This function loads the entire file into memory.
    It is intended for configuration files, not large datasets.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
        TypeError: If the YAML content is not a dictionary.
    """
    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        msg = f"YAML file {path} must contain a dictionary, got {type(data)}"
        raise TypeError(msg)

    return data

def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """
    Dump data to a YAML file.

    Args:
        data: Dictionary to dump.
        path: Path to the output file.
    """
    with path.open("w") as f:
        yaml.dump(data, f, sort_keys=False)


def run_subprocess(
    command: list[str],
    cwd: Path | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """
    Run a subprocess command safely.

    Args:
        command: List of command arguments.
        cwd: Working directory.
        timeout: Timeout in seconds.
        env: Environment variables.

    Returns:
        Tuple of (return_code, stdout, stderr).
    """
    try:
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            env=env,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
        # We return a synthetic error code for timeout, or raise?
        # The Spec says: "Timeout: Mock a timeout in subprocess. Assert status=TIMEOUT."
        # If I raise TimeoutExpired, the caller needs to handle it.
        # It's better to raise so the caller can distinguish timeout from crash.
        raise e
    except Exception as e:
        logger.exception(f"Subprocess failed: {e}")
        raise e
