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

def initialize_project(config_file: Path) -> Tuple[SystemConfig, DatabaseManager]:
    """
    Initializes the project environment.

    1. Loads and validates configuration.
    2. Sets up logging.
    3. Initializes the database.

    Args:
        config_file: Path to the input YAML configuration.

    Returns:
        A tuple containing the initialized SystemConfig and DatabaseManager.
    """
    # 1. Config Factory: Load and Expand
    config = ConfigFactory.from_yaml(config_file)

    # 2. Setup Logging
    setup_logging(config.log_path)
    logging.info("System logging initialized.")

    # 3. Initialize Database
    db = DatabaseManager(config.db_path)
    db.initialize(config)
    logging.info(f"System initialized. DB: {config.db_path}")

    return config, db
