from pathlib import Path
from mlip_autopipec.core.config import load_config

def validate_config_file(config_path: Path) -> str:
    """
    Validates the configuration file at the given path.

    Args:
        config_path: Path to the configuration file.

    Returns:
        "OK" if validation is successful.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the configuration is invalid.
        Exception: For other errors.
    """
    if not config_path.exists():
        msg = f"Config file {config_path} not found."
        raise FileNotFoundError(msg)

    load_config(config_path)
    return "OK"
