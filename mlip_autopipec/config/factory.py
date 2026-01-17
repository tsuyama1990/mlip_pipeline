from pathlib import Path

import yaml

from mlip_autopipec.config.models import MinimalConfig, SystemConfig


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

        with path.open("r") as f:
            data = yaml.safe_load(f)

        minimal = MinimalConfig.model_validate(data)

        # Resolve paths
        # Working dir is created in the current working directory under project_name
        # Or should it be where the input file is? UAT says "in the current path".
        # We assume current working directory of the process.
        cwd = Path.cwd()
        working_dir = cwd / minimal.project_name

        # Create directory structure
        working_dir.mkdir(parents=True, exist_ok=True)

        db_path = working_dir / "project.db"
        log_path = working_dir / "system.log"

        return SystemConfig(
            minimal=minimal,
            working_dir=working_dir,
            db_path=db_path,
            log_path=log_path
        )
