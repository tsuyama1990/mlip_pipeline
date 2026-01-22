"""
MLIP-AutoPipe: Automated Machine Learning Interatomic Potential Pipeline.

Architecture Overview:
----------------------
This package is organized into several key modules that interact to produce
machine learning potentials.

1. **Config (`mlip_autopipec.config`)**:
   Defines strict Pydantic schemas for all inputs (`DFTConfig`, `GeneratorConfig`, etc.).
   This is the source of truth for the pipeline.

2. **Core (`mlip_autopipec.core`)**:
   Provides foundational services like `DatabaseManager` (for SQLite/ASE access)
   and logging.

3. **Generator (`mlip_autopipec.generator`)**:
   Responsible for creating atomic structures. `StructureBuilder` orchestrates
   strategies like SQS (alloys) and Defects to populate the database.

4. **Surrogate (`mlip_autopipec.surrogate`)**:
   Implements the Active Learning loop. `SurrogatePipeline` uses a foundation model
   (MACE) to screen candidates and `FarthestPointSampling` to select diverse ones.

5. **DFT (`mlip_autopipec.dft`)**:
   The "Factory" for ground-truth data. `QERunner` executes Quantum Espresso
   calculations, handling errors via `RecoveryHandler` and parsing results
   with `QEOutputParser`.

6. **Orchestration (`mlip_autopipec.orchestration`)**:
   `WorkflowManager` ties everything together, managing state transitions
   (Pending -> Selected -> Completed) and distributing tasks via `TaskQueue`.

Interaction Flow:
-----------------
User Config -> WorkflowManager -> Generator -> DB -> Surrogate -> DB -> DFT -> DB -> Trainer
"""

from mlip_autopipec.config.models import (
    DFTConfig,
    GeneratorConfig,
    MLIPConfig,
    SurrogateConfig,
    SystemConfig,
    TargetSystem,
)
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.task_queue import TaskQueue
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline

__all__ = [
    "DFTConfig",
    "DatabaseManager",
    "GeneratorConfig",
    "MLIPConfig",
    "QERunner",
    "StructureBuilder",
    "SurrogateConfig",
    "SurrogatePipeline",
    "SystemConfig",
    "TargetSystem",
    "TaskQueue",
    "WorkflowManager",
]
