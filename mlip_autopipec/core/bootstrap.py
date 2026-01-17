"""
Bootstrap utilities for application startup.
"""
from pathlib import Path
from typing import Tuple
import logging

from mlip_autopipec.config.factory import ConfigFactory
from mlip_autopipec.config.schemas.system import SystemConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.core.logging import setup_logging
from mlip_autopipec.core.workspace import WorkspaceManager

def initialize_project(config_file: Path) -> Tuple[SystemConfig, DatabaseManager]:
    """
    Initializes the project environment.

    Orchestration Sequence:
    1. Load and validate configuration (ConfigFactory).
    2. Setup Workspace (WorkspaceManager).
    3. Setup Logging (logging).
    4. Initialize Database (DatabaseManager).

    Args:
        config_file: Path to the input YAML configuration.

    Returns:
        A tuple containing the initialized SystemConfig and DatabaseManager.

    Raises:
        ConfigError: If configuration is invalid.
        WorkspaceError: If directories cannot be created.
        LoggingError: If logging cannot be set up.
        DatabaseError: If database cannot be initialized.
    """
    # 1. Config Factory: Load and Expand (Pure)
    config = ConfigFactory.from_yaml(config_file)

    # 2. Setup Workspace (Side Effects)
    WorkspaceManager.setup_workspace(config)

    # 3. Setup Logging (Side Effects)
    setup_logging(config.log_path)
    logging.info("System logging initialized.")

    # 4. Initialize Database (Side Effects)
    db = DatabaseManager(config.db_path)
    db.initialize(config)
    logging.info(f"System initialized. DB: {config.db_path}")

    return config, db
