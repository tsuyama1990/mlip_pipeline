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

Module Interactions:
--------------------
The system follows a linear pipeline architecture orchestrated by `WorkflowManager`:

1.  **Initialization**: `WorkflowManager` loads `MLIPConfig` and initializes the `DatabaseManager`.
2.  **Generation**:
    - `StructureBuilder` generates structures based on `GeneratorConfig`.
    - Structures are saved to DB with status "pending".
3.  **Surrogate Selection**:
    - `SurrogatePipeline` queries "pending" structures.
    - `MaceWrapper` filters unphysical structures (force threshold).
    - `FarthestPointSampling` selects diverse subset.
    - Selected structures marked "selected"; others "held" or "rejected".
4.  **DFT Execution**:
    - `TaskQueue` picks up "selected" structures.
    - `QERunner` executes Quantum Espresso.
    - On success: Results (Energy/Forces/Stress) saved to DB, status -> "completed".
    - On failure: `RecoveryHandler` attempts parameter adjustments (beta, smearing). If all fail, status -> "failed".
5.  **Training** (Future):
    - `Pacemaker` trains on "completed" structures.

Dependencies:
-------------
- `config` -> All modules (schemas)
- `core` -> All modules (database/logging)
- `orchestration` -> `generator`, `surrogate`, `dft` (control flow)
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
