from pathlib import Path

import yaml

from mlip_autopipec.config.models import MinimalConfig, SystemConfig


class ConfigFactory:
    @staticmethod
    def from_yaml(path: Path) -> SystemConfig:
        path = Path(path)
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("r") as f:
            data = yaml.safe_load(f)

        minimal = MinimalConfig.model_validate(data)

        # Determine working directory
        # The user provides a project name. We create a directory with that name
        # in the current working directory.
        # Alternatively, we could allow the user to specify a base path, but for now
        # consistent with SPEC: "creates directory structure (e.g., workspace/run_01/)"
        # But SPEC also says "The user provides a project_name" and UAT says
        # "directory named AlCu_Alloy should be created in the current path".

        working_dir = Path.cwd() / minimal.project_name
        working_dir.mkdir(parents=True, exist_ok=True)

        # SPEC says: "Automatically sets the db_path to {project_name}.db based on the user input"
        # The ConfigFactory automatically sets the db_path to {project_name}.db
        # based on the user input, overriding the default mlip_database.db defined in SystemConfig.
        # "db_path must be absolute"

        db_path = working_dir / f"{minimal.project_name}.db"
        log_path = working_dir / "system.log"

        return SystemConfig(
            minimal=minimal,
            working_dir=working_dir.resolve(),
            db_path=db_path.resolve(),
            log_path=log_path.resolve(),
        )
