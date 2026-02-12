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


class EnvVarExpander:
    """
    Stream wrapper that expands environment variables on the fly.

    This class wraps a text stream and performs regex substitution
    on the read content. To handle environment variables split across
    read chunks, it maintains a buffer.
    """

    def __init__(self, stream: IO[str]) -> None:
        self.stream = stream
        self.pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")
        self.buffer = ""
        # We need a way to know if we are at EOF of source
        self.eof = False

    def read(self, size: int = -1) -> str:
        if size == -1:
            # Read all remaining
            content = self.buffer + self.stream.read()
            self.buffer = ""
            return self._expand(content)

        if size == 0:
            return ""

        # We need to ensure we return roughly 'size' bytes,
        # but we must ensure we don't split a ${VAR} token.
        # Strategy: Read 'size' + extra from stream.
        # Append to buffer.
        # Find the last safe position (not inside a potential ${...}).
        # Expand up to that position.
        # Keep the rest in buffer.

        # Heuristic: Read size.
        chunk = self.stream.read(size)
        if not chunk:
            self.eof = True

        self.buffer += chunk

        # If buffer is empty (and EOF), return empty
        if not self.buffer:
            return ""

        # If we hit EOF, we must process everything in buffer
        if self.eof:
            expanded = self._expand(self.buffer)
            self.buffer = ""
            return expanded

        # Check for potential partial tokens at the end of buffer.
        # A partial token starts with $
        # and might be incomplete.
        # We search for the last '$'.
        last_dollar = self.buffer.rfind('$')

        if last_dollar == -1:
            # Safe to expand everything
            to_process = self.buffer
            self.buffer = ""
        else:
            # Check if it looks like an incomplete var
            # Case 1: $ at end
            # Case 2: ${ at end
            # Case 3: ${VA at end
            # We treat everything from the last $ as potentially incomplete
            # UNLESS it's clearly complete (e.g. $VAR followed by non-var char, or ${VAR})
            # Simplifying: Just keep the part from last '$' in buffer
            # unless it's way too long (avoid buffer growth exploit).
            # But wait, what if we have multiple $?
            # We process up to last_dollar.

            # Optimization: If last_dollar is very old (buffer large), force process?
            # For config parsing, vars are usually short.

            to_process = self.buffer[:last_dollar]
            self.buffer = self.buffer[last_dollar:]

            # Edge case: What if buffer contains ONLY a partial token?
            # e.g. we read "$". We put "$" in buffer. to_process is empty.
            # We return empty. Next read calls. We assume yaml parser handles short reads (it does).

            # But what if we return empty string? Does loader think EOF?
            # PyYAML's reader checks for empty string.
            # If we return empty but not EOF, we might hang or error.
            # We MUST return at least 1 char if available, unless EOF.

            if not to_process and self.buffer:
                # We have only potential partial token in buffer.
                # We need to read more to complete it or determine it's not a token.
                # Recursive call with small size?
                # Or just read more here.
                more = self.stream.read(100) # Read a bit more
                if not more:
                    self.eof = True
                    # Process buffer as is
                    to_process = self.buffer
                    self.buffer = ""
                else:
                    self.buffer += more
                    # Recurse logic? Or loop?
                    # Let's simplify: return self.read(size) again (it will hit the top logic)
                    return self.read(size)

        return self._expand(to_process)

    def _expand(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, "")
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
            stream = EnvVarExpander(f)
            data = yaml.safe_load(stream)

        if not isinstance(data, dict):
            msg = "Configuration file must parse to a dictionary."
            raise ConfigError(msg)

        return GlobalConfig.model_validate(data)

    except (yaml.YAMLError, OSError) as e:
        msg = f"Error loading config: {e}"
        raise ConfigError(msg) from e
    except ValidationError as e:
        msg = f"Configuration validation failed: {e}"
        raise ConfigError(msg) from e
