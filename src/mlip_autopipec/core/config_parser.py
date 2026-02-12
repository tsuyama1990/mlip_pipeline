import os
import re
from pathlib import Path
from typing import Any

import yaml

from mlip_autopipec.domain_models.config import FullConfig

# Regex to match ${VAR} or $VAR
# Groups: 1=${VAR}, 2=$VAR
ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")


def _substitute_env(value: str) -> str:
    """
    Substitute environment variables in a string.
    Raises ValueError if a variable is not found in os.environ.
    """

    def repl(match: re.Match[str]) -> str:
        var = match.group(1) or match.group(2)
        if var not in os.environ:
            msg = f"Environment variable '{var}' not set"
            raise ValueError(msg)
        return os.environ[var]

    return ENV_PATTERN.sub(repl, value)


def _walk_and_substitute(data: Any) -> Any:
    """
    Recursively traverse data structure and substitute environment variables in strings.
    """
    if isinstance(data, dict):
        return {k: _walk_and_substitute(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_walk_and_substitute(v) for v in data]
    if isinstance(data, str):
        return _substitute_env(data)
    return data


def load_config(path: Path) -> FullConfig:
    """
    Load configuration from a YAML file, substitute environment variables,
    and validate against FullConfig schema.
    """
    with path.open("r") as f:
        raw_data = yaml.safe_load(f)

    data = _walk_and_substitute(raw_data)
    return FullConfig.model_validate(data)
