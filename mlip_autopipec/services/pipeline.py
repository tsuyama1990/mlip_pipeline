"""
This module contains the `PipelineController` which orchestrates the
MLIP-AutoPipe workflow execution.
"""

import logging
from pathlib import Path

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.loaders.yaml_loader import ConfigLoader
from mlip_autopipec.workflow_manager import WorkflowManager

log = logging.getLogger(__name__)


class PipelineController:
    """
    Orchestrates the loading, configuration, and execution of the workflow.
    """

    @staticmethod
    def execute(config_file: Path) -> None:
        """
        Executes the full pipeline given a configuration file path.

        Args:
            config_file: Path to the user's YAML configuration.

        Raises:
            Exception: Propagates exceptions for CLI handling.
        """
        log.info(f"Starting pipeline execution with config: {config_file}")

        # 1. Load User Config
        user_config = ConfigLoader.load_user_config(config_file)
        log.info(f"Loaded configuration for project: {user_config.project_name}")

        # 2. Create System Config
        system_config = ConfigFactory.from_user_input(user_config)
        log.info(f"System configuration generated. Run UUID: {system_config.run_uuid}")

        # 3. Initialize Workflow
        # Using current working directory for execution context
        work_dir = Path.cwd()
        manager = WorkflowManager(system_config=system_config, work_dir=work_dir)

        # 4. Run Workflow
        log.info("Dispatching workflow execution to manager.")
        manager.run()
        log.info("Workflow execution completed successfully.")
