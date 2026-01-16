from pathlib import Path

import yaml

from mlip_autopipec.config.models import UserInputConfig


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
            yaml.YAMLError: If YAML parsing fails.
            ValidationError: If Pydantic validation fails.
        """
        with config_path.open('r') as f:
            raw_data = yaml.safe_load(f)
        return UserInputConfig.model_validate(raw_data)
