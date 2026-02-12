import os
import re
from pathlib import Path
from typing import IO, Any

import yaml
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig


class ConfigError(Exception):
    """Raised when configuration loading fails."""


# Max config file size: 1MB
MAX_CONFIG_SIZE = 1 * 1024 * 1024


def _load_yaml_with_env_vars(stream: IO[str]) -> Any:
    """
    Loads YAML from a stream, expanding environment variables.

    This function reads the stream line by line to perform expansion
    before parsing, avoiding loading the entire file into a single string if possible,
    though yaml.safe_load will eventually construct the full object graph in memory.
    The primary goal here is to avoid the intermediate *string* representation of the whole file.
    """
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")

    def expand_line(line: str) -> str:
        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, "")
        return pattern.sub(repl, line)

    # Generator that yields expanded lines
    def line_generator() -> Any:
        for line in stream:
            yield expand_line(line)

    # yaml.safe_load accepts a stream or string.
    # However, it doesn't accept a generator directly.
    # We can wrap the generator in a class that behaves like a stream (read/readline).

    # We consume the generator into a string.
    # Given the strict file size limit (1MB), this is safe and efficient enough.
    # Implementing a true streaming reader for yaml.safe_load that handles env var expansion
    # on the fly would require a custom Loader which is error-prone.
    return yaml.safe_load("".join(line_generator()))


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
        with config_path.open("r", encoding="utf-8") as f:
            data = _load_yaml_with_env_vars(f)

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
