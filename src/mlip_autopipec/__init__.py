"""
MLIP-AutoPipe: Automated Machine Learning Interatomic Potential Pipeline.

This package provides a zero-human automated pipeline for generating, labeling (DFT),
training, and verifying machine learning interatomic potentials.

Key Components
--------------
- **Configuration**: Strictly typed Pydantic models (see `mlip_autopipec.config`).
- **Orchestration**: `WorkflowManager` coordinates the active learning loop.
- **Database**: `DatabaseManager` handles all persistence via ASE/SQLite.

Usage
-----
The primary interface is via the Command Line Interface (CLI):
    $ mlip-auto --help

For programmatic usage, initialize a configuration and run the workflow:

    from mlip_autopipec import MLIPConfig, WorkflowManager

    config = MLIPConfig(...)
    manager = WorkflowManager(config)
    manager.run()

"""

from mlip_autopipec.config.models import (
    MLIPConfig,
    SystemConfig,
)
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.orchestration.workflow import WorkflowManager

__all__ = [
    "DatabaseManager",
    "MLIPConfig",
    "SystemConfig",
    "WorkflowManager",
]
