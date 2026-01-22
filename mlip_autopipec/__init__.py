"""
MLIP-AutoPipe: Automated Machine Learning Interatomic Potential Pipeline.

Architecture Overview
---------------------
This package is structured into modular components following a Hexagonal (Ports & Adapters) style architecture,
orchestrated by a central Workflow Manager.

Component Interaction Diagram:
------------------------------

    [User Config (YAML)]
            |
            v
    +-----------------------+
    |   Workflow Manager    | <---- [Task Queue (Dask)]
    |   (State Machine)     |
    +----------+------------+
               |
               v
    +----------+----------+      +------------------+
    |   Phase Executor    | <--> | Database Manager | <--> [SQLite (ase.db)]
    +----------+----------+      +------------------+
               |
               +---> [Generator] (StructureBuilder) -> Create Candidates
               |
               +---> [Surrogate] (SurrogatePipeline) -> Screen & Select
               |
               +---> [DFT Factory] (QERunner) -> Label Data (Quantum Espresso)
               |
               +---> [Training] (PacemakerWrapper) -> Fit Potential (Pacemaker)
               |
               +---> [Inference] (LammpsRunner) -> Run MD & UQ (LAMMPS)
                                        |
                                        v
                                 [EmbeddingExtractor] -> Active Learning Feedback

Modules & Interactions:

1.  **Config (`mlip_autopipec.config`)**
    -   **Role**: Source of Truth.
    -   **Interactions**: Imported by all other modules. Validates inputs before any execution begins.

2.  **Core (`mlip_autopipec.core`)**
    -   **Role**: Infrastructure Layer.
    -   **Database**: `DatabaseManager` wraps `ase.db` (SQLite). It provides atomic CRUD operations.
        It is used by Generator (write), Surrogate (read/update), and DFT (read/update).
    -   **Logging**: Centralized logging configuration.

3.  **Generator (`mlip_autopipec.generator`)**
    -   **Role**: Domain Logic for creating atomic structures.
    -   **Interactions**: `StructureBuilder` receives `GeneratorConfig`, applies strategies (SQS, Defects),
        and writes new "pending" candidates to the Database.

4.  **Surrogate (`mlip_autopipec.surrogate`)**
    -   **Role**: Active Learning Strategy.
    -   **Interactions**: `SurrogatePipeline` reads "pending" structures from DB. It delegates to
        `MaceWrapper` (Adapter) for inference and `FarthestPointSampling` for selection.
        Updates DB status to "selected" or "rejected".

5.  **DFT (`mlip_autopipec.dft`)**
    -   **Role**: Ground Truth Factory.
    -   **Interactions**: `QERunner` reads "selected" structures. It executes Quantum Espresso via subprocess.
        `RecoveryHandler` interprets errors. `QEOutputParser` extracts results.
        Updates DB status to "completed".

6.  **Training (`mlip_autopipec.training`)**
    -   **Role**: Model Fitting.
    -   **Interactions**: `PacemakerWrapper` exports data from DB to `.xyz`, generates `input.yaml`,
        and runs `pacemaker` binary. Produces `.yace` potential files.

7.  **Inference (`mlip_autopipec.inference`)**
    -   **Role**: Simulation & Feedback.
    -   **Interactions**: `LammpsRunner` executes MD using the trained potential. `ScriptGenerator` creates
        LAMMPS scripts with `fix halt` for uncertainty termination. `EmbeddingExtractor` creates new
        cluster candidates from high-uncertainty frames.

8.  **Orchestration (`mlip_autopipec.orchestration`)**
    -   **Role**: Control Plane.
    -   **Interactions**: `WorkflowManager` monitors DB state and triggers the appropriate module via `PhaseExecutor`.
        `TaskQueue` (Adapter) handles distributed execution logic (Dask).

Key Data Flow:
[User Config] -> [StructureBuilder] -> [Database (Pending)] -> [SurrogatePipeline] -> [Database (Selected)] -> [QERunner] -> [Database (Completed)] -> [Pacemaker] -> [Potential] -> [LAMMPS] -> [Database (Active Learning)]
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
from mlip_autopipec.orchestration.task_queue import TaskQueue
from mlip_autopipec.orchestration.workflow import WorkflowManager
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
