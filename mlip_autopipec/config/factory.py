import yaml
from pathlib import Path
from typing import Any

from mlip_autopipec.config.schemas.common import MinimalConfig
from mlip_autopipec.config.schemas.system import SystemConfig

class ConfigFactory:
    """
    A factory for creating application configurations.
    Responsible for reading User Input, validating it, and setting up the System Environment.
    """

    @staticmethod
    def from_yaml(path: Path) -> SystemConfig:
        """
        Reads a YAML file, validates it against MinimalConfig,
        creates the project directory structure, and returns the SystemConfig.
        """
        # Resolve input file path
        input_path = Path(path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {input_path}")

        # Load YAML
        with input_path.open('r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}") from e

        # Validate User Input
        try:
            minimal = MinimalConfig.model_validate(data)
        except Exception as e:
            # Wrap validation error for better CLI handling if needed
            # But usually Pydantic ValidationError is good enough
            raise e

        # Determine Paths
        # Create project directory in the current working directory
        cwd = Path.cwd()
        working_dir = cwd / minimal.project_name

        # Create directory structure
        working_dir.mkdir(parents=True, exist_ok=True)

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
