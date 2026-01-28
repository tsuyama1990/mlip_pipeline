import logging
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from mlip_autopipec.config.models import UserInputConfig
from mlip_autopipec.config.schemas.dft import DFTConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ConfigLoader:
    """Loads and validates user configuration."""

    @staticmethod
    def load_user_config(config_path: Path) -> UserInputConfig:
        """
        Reads a YAML file and validates it against UserInputConfig.
        """
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        try:
            with config_path.open("r") as f:
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


def load_config(path: str | Path, model: type[T]) -> T:
    """
    Load a YAML configuration file and validate it against a Pydantic model.
    Adapter for simple usage.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = f"Config file {path} must contain a dictionary"
        raise ValueError(msg)

    return model(**data)


def load_dft_config_helper(path: Path) -> tuple[DFTConfig, Path]:
    """
    Helper to load DFT config from full MLIP config or standalone DFT config.
    Returns: (DFTConfig, work_dir)
    """
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = "Config must be a dictionary"
        raise ValueError(msg)

    work_dir = Path("dft_work")  # Default

    if "dft" in data:
        # It's likely an MLIPConfig
        dft_config = DFTConfig(**data["dft"])
        if "runtime" in data and "work_dir" in data["runtime"]:
            work_dir = Path(data["runtime"]["work_dir"])
    else:
        # Assume standalone
        dft_config = DFTConfig(**data)

    return dft_config, work_dir
