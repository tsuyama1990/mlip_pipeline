"""
This module provides a factory class for creating the comprehensive
SystemConfig from a high-level UserInputConfig.
"""
import yaml
from pathlib import Path

from mlip_autopipec.config.schemas.common import MinimalConfig
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.exceptions import ConfigError

class ConfigFactory:
    """
    A factory for creating application configurations.
    Responsible for reading User Input, validating it, and setting up the System Environment.
    """

    @staticmethod
    def from_yaml(path: Path) -> SystemConfig:
        """
        Reads a YAML file, validates it against MinimalConfig,
        and returns the SystemConfig with resolved paths.

        This method is PURE: it does not create directories on disk.

        Args:
            path: Path to the input YAML file.

        Returns:
            A fully populated SystemConfig object.

        Raises:
            ConfigError: If the file is missing, invalid YAML, or validation fails.
        """
        # Resolve input file path
        try:
            input_path = Path(path).resolve()
        except Exception as e:
             raise ConfigError(f"Invalid path provided: {path}") from e

        if not input_path.exists():
            raise ConfigError(f"Configuration file not found: {input_path}")

        # Load YAML
        try:
            with input_path.open('r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML format in {input_path}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Failed to read {input_path}: {e}") from e

        # Validate User Input
        try:
            minimal = MinimalConfig.model_validate(data)
        except Exception as e:
             raise ConfigError(f"Configuration validation failed: {e}") from e

        # Determine Paths
        # Create project directory in the current working directory
        cwd = Path.cwd()
        working_dir = cwd / minimal.project_name

        # Define internal paths
        db_path = working_dir / "project.db"
        log_path = working_dir / "system.log"

        # Create SystemConfig
        system_config = SystemConfig(
            minimal=minimal,
            working_dir=working_dir,
            db_path=db_path,
            log_path=log_path
        )

        return system_config
