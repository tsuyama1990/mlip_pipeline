"""
Pipeline Service to orchestrate the initialization and execution of the MLIP pipeline.
This layer separates business logic from the CLI (app.py).
"""

import logging
from pathlib import Path

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.core.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

class PipelineController:
    """
    Controller for the MLIP pipeline execution.
    """

    @staticmethod
    def execute(input_file: Path) -> SystemConfig:
        """
        Orchestrates the initialization of the pipeline.

        Args:
            input_file: Path to the user configuration file (YAML).

        Returns:
            The initialized SystemConfig.
        """
        # 1. Load Config
        config = ConfigFactory.from_yaml(input_file)

        # 2. Setup Workspace
        workspace = WorkspaceManager(config)
        workspace.setup_workspace()

        # 3. Setup Logging
        setup_logging(config.log_path)

        # 4. Initialize Database
        db = DatabaseManager(config.db_path)
        db.initialize()
        db.set_system_config(config)

        logger.info("System initialized via PipelineController")
        return config
