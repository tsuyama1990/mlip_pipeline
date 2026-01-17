from pathlib import Path

import yaml
from yaml.error import YAMLError

from mlip_autopipec.config.models import MinimalConfig, SystemConfig
from mlip_autopipec.exceptions import ConfigError


class ConfigFactory:
    """
    Factory for creating SystemConfig from user input.
    """

    @staticmethod
    def from_yaml(path: Path) -> SystemConfig:
        """
        Reads a YAML file, validates it against MinimalConfig,
        and returns a fully hydrated SystemConfig.
        """
        path = Path(path).resolve()

        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            with path.open("r") as f:
                data = yaml.safe_load(f)
        except YAMLError as e:
            msg = f"Failed to parse YAML configuration: {e}"
            raise ConfigError(msg) from e
        except OSError as e:
            msg = f"Failed to read configuration file: {e}"
            raise ConfigError(msg) from e

        try:
            minimal = MinimalConfig.model_validate(data)
        except Exception as e:
             # Pydantic validation errors are detailed, re-raising them as is usually better
             # or wrapping them. The user wants specific error messages.
             # Pydantic ValidationError is a safe exception to propagate or wrap.
             # We will wrap it if we want a unified ConfigError, but keeping the detail is key.
             # Let's wrap but ensure detail is preserved.
             msg = f"Configuration validation failed: {e}"
             raise ConfigError(msg) from e

        # Resolve paths
        # We assume current working directory of the process.
        cwd = Path.cwd()
        working_dir = cwd / minimal.project_name

        db_path = working_dir / "project.db"
        log_path = working_dir / "system.log"

        return SystemConfig(
            minimal=minimal,
            working_dir=working_dir,
            db_path=db_path,
            log_path=log_path
        )
