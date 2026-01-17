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
            # We strictly read the file content as text first
            file_content = path.read_text(encoding="utf-8")
            # Basic sanity check to avoid processing massive files
            if len(file_content) > 10 * 1024 * 1024: # 10MB limit
                msg = "Configuration file too large."
                raise ConfigError(msg)

            data = yaml.safe_load(file_content)

            if not isinstance(data, dict):
                msg = "Configuration file must contain a YAML dictionary."
                raise ConfigError(msg)

        except YAMLError as e:
            msg = f"Failed to parse YAML configuration: {e}"
            raise ConfigError(msg) from e
        except OSError as e:
            msg = f"Failed to read configuration file: {e}"
            raise ConfigError(msg) from e

        try:
            minimal = MinimalConfig.model_validate(data)
        except Exception as e:
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
