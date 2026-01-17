import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from mlip_autopipec.config.models import UserInputConfig

log = logging.getLogger(__name__)

class ConfigLoader:
    """Loads and validates user configuration."""

    @staticmethod
    def load_user_config(config_path: Path) -> UserInputConfig:
        """
        Reads a YAML file and validates it against UserInputConfig.

        Args:
            config_path: Path to the YAML file.

        Returns:
            Validated UserInputConfig object.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If YAML parsing fails or data is invalid.
        """
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        try:
            with config_path.open('r') as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            log.exception(f"Failed to parse YAML file: {config_path}")
            msg = f"Invalid YAML format in {config_path}"
            raise ValueError(msg) from e

        if not isinstance(raw_data, dict):
            msg = f"Configuration file {config_path} must contain a dictionary/mapping."
            raise ValueError(msg)

        try:
            return UserInputConfig.model_validate(raw_data)
        except ValidationError as e:
            log.exception(f"Configuration validation failed for {config_path}")
            msg = f"Configuration validation failed: {e}"
            raise ValueError(msg) from e
