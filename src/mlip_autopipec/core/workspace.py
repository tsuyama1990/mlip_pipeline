from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.exceptions import WorkspaceError


class WorkspaceManager:
    """
    Manages the physical workspace (directories, files) for the project.
    Separates filesystem side-effects from configuration logic.
    """

    def __init__(self, system_config: SystemConfig) -> None:
        self.config = system_config

    def setup_workspace(self) -> None:
        """
        Creates the directory structure defined in the SystemConfig.
        """
        try:
            self.config.working_dir.mkdir(parents=True, exist_ok=True)
            # Create parent directories for log and db if they are disjoint
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Failed to create workspace directories: {e}"
            raise WorkspaceError(msg) from e
