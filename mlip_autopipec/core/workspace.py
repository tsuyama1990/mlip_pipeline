"""
Utilities for managing the project workspace on the filesystem.
"""
from pathlib import Path
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.exceptions import WorkspaceError

class WorkspaceManager:
    """
    Manages the creation and verification of the project directory structure.
    """

    @staticmethod
    def setup_workspace(config: SystemConfig) -> None:
        """
        Creates the necessary directories for the project.

        Args:
            config: The system configuration containing the paths.

        Raises:
            WorkspaceError: If directory creation fails.
        """
        try:
            # Create the main working directory
            config.working_dir.mkdir(parents=True, exist_ok=True)

            # Create parent directories for DB and Log if they differ (unlikely in current schema but safe)
            config.db_path.parent.mkdir(parents=True, exist_ok=True)
            config.log_path.parent.mkdir(parents=True, exist_ok=True)

        except OSError as e:
            raise WorkspaceError(f"Failed to create workspace directories at {config.working_dir}: {e}") from e
