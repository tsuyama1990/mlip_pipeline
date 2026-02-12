import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig


class ConfigError(Exception):
    """Raised when configuration loading fails."""


# Max config file size: 1MB
MAX_CONFIG_SIZE = 1 * 1024 * 1024


def _expand_vars(content: str) -> str:
    """
    Expands environment variables in the format ${VAR} or $VAR.

    Args:
        content: The YAML content as a string.

    Returns:
        The content with environment variables expanded.
    """
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")

    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, "")

    return pattern.sub(repl, content)


def load_config(config_path: Path) -> GlobalConfig:
    """
    Loads and validates the configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated GlobalConfig object.

    Raises:
        ConfigError: If the file is missing or invalid.
    """
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise ConfigError(msg)

    # Check file size
    if config_path.stat().st_size > MAX_CONFIG_SIZE:
        msg = f"Configuration file too large (max {MAX_CONFIG_SIZE} bytes)"
        raise ConfigError(msg)

    try:
        content = config_path.read_text(encoding="utf-8")
        expanded_content = _expand_vars(content)
        data = yaml.safe_load(expanded_content)

        if not isinstance(data, dict):
            msg = "Configuration file must parse to a dictionary."
            raise ConfigError(msg)  # noqa: TRY301

        return GlobalConfig.model_validate(data)

    except yaml.YAMLError as e:
        msg = f"Error parsing YAML: {e}"
        raise ConfigError(msg) from e
    except ValidationError as e:
        msg = f"Configuration validation failed: {e}"
        raise ConfigError(msg) from e
    except Exception as e:
        msg = f"Unexpected error loading config: {e}"
        raise ConfigError(msg) from e
