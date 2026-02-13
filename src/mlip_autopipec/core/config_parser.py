import os
import re
from pathlib import Path
from typing import IO

import yaml
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig


class ConfigError(Exception):
    """Raised when configuration loading fails."""


# Max config file size: 1MB
MAX_CONFIG_SIZE = 1 * 1024 * 1024
# Max buffer size during expansion
MAX_BUFFER_SIZE = 64 * 1024


class EnvVarExpander:
    """
    Stream wrapper that expands environment variables on the fly.

    This class wraps a text stream and performs regex substitution
    on the read content. To handle environment variables split across
    read chunks, it maintains a buffer.
    """

    def __init__(self, stream: IO[str], strict: bool = True) -> None:
        self.stream = stream
        self.pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")
        self.buffer = ""
        # We need a way to know if we are at EOF of source
        self.eof = False
        self.strict = strict
        self.total_read = 0

    def read(self, size: int = -1) -> str:
        # Enforce max read limit to prevent memory bombs
        if self.total_read > MAX_CONFIG_SIZE:
            msg = f"Configuration exceeded max size ({MAX_CONFIG_SIZE} bytes) during expansion."
            raise ConfigError(msg)

        if size == -1:
            # Read all remaining
            content = self.buffer + self.stream.read()
            self.buffer = ""
            self.total_read += len(content)
            return self._expand(content)

        if size == 0:
            return ""

        # Read logic extracted to helper
        return self._read_chunk(size)

    def _read_chunk(self, size: int) -> str:
        # Heuristic: Read size.
        chunk = self.stream.read(size)
        if not chunk:
            self.eof = True

        # Check buffer size to prevent memory exhaustion
        if len(self.buffer) + len(chunk) > MAX_BUFFER_SIZE:
            msg = f"Expansion buffer exceeded max size ({MAX_BUFFER_SIZE} bytes). Possible incomplete variable token."
            raise ConfigError(msg)

        self.buffer += chunk

        # If buffer is empty (and EOF), return empty
        if not self.buffer:
            return ""

        # If we hit EOF, we must process everything in buffer
        if self.eof:
            expanded = self._expand(self.buffer)
            self.buffer = ""
            self.total_read += len(expanded)
            return expanded

        return self._process_buffer(size)

    def _process_buffer(self, size: int) -> str:
        # Check for potential partial tokens at the end of buffer.
        last_dollar = self.buffer.rfind("$")

        if last_dollar == -1:
            # Safe to expand everything
            to_process = self.buffer
            self.buffer = ""
        else:
            # Found a dollar sign, potentially start of a variable
            to_process = self.buffer[:last_dollar]
            self.buffer = self.buffer[last_dollar:]

            if not to_process and self.buffer:
                # Only potential partial token in buffer.
                # Need to read more.
                more = self.stream.read(100)  # Read a bit more
                if not more:
                    self.eof = True
                    to_process = self.buffer
                    self.buffer = ""
                else:
                    if len(self.buffer) + len(more) > MAX_BUFFER_SIZE:
                        msg = f"Expansion buffer exceeded max size ({MAX_BUFFER_SIZE} bytes)."
                        raise ConfigError(msg)
                    self.buffer += more
                    # Recurse logic via read(size) which calls _read_chunk again
                    # But _read_chunk calls stream.read(size), we want to just process buffer again?
                    # Actually, calling read(size) again is fine, it will try to read from stream
                    # but stream pointer moved.
                    # Wait, if we recurse, read(size) reads MORE from stream.
                    # We want to re-evaluate buffer.
                    # Let's just call _process_buffer recursively?
                    # No, _process_buffer expects logic to split.
                    # If we added 'more' to buffer, last_dollar position might change (if 'more' has no $).
                    # Actually, if we added 'more', we should loop back.
                    # Recursion via read(size) is simplest but might over-read.
                    # Let's allow recursion.
                    return self.read(size)

        expanded = self._expand(to_process)
        self.total_read += len(expanded)
        return expanded

    def _expand(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1) or match.group(2)
            val = os.environ.get(var_name)
            if val is None:
                if self.strict:
                    msg = f"Missing environment variable: {var_name}"
                    raise ConfigError(msg)
                return ""
            return val

        return self.pattern.sub(repl, text)


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
            # Use EnvVarExpander to stream content
            # strict=True enforces required env vars
            stream = EnvVarExpander(f, strict=True)
            data = yaml.safe_load(stream)

        if not isinstance(data, dict):
            msg = "Configuration file must parse to a dictionary."
            raise ConfigError(msg)

        # Basic protection against Billion Laughs expansion if it resulted in huge dict
        if len(data) > 1000:
            msg = "Configuration file contains too many keys."
            raise ConfigError(msg)

        return GlobalConfig.model_validate(data)

    except (yaml.YAMLError, OSError) as e:
        msg = f"Error loading config: {e}"
        raise ConfigError(msg) from e
    except ValidationError as e:
        msg = f"Configuration validation failed: {e}"
        raise ConfigError(msg) from e
