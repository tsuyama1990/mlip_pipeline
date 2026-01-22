# System Architecture

## Overview

MLIP-AutoPipe is a modular, event-driven system designed to automate the generation, calculation, and active learning of machine learning interatomic potentials.

## Diagram

```mermaid
graph TD
    User[User Config] --> Manager[WorkflowManager]

    subgraph "Orchestration Layer"
        Manager --> DB[(DatabaseManager)]
        Manager --> TQ[TaskQueue (Dask)]
        Manager --> Dash[Dashboard]
    end

    subgraph "Execution Layer"
        Manager --> Gen[StructureBuilder (Generator)]
        Gen --> SQS[SQS Strategy]
        Gen --> Dist[Distortions]
        Gen --> Defect[Defects]

        Manager --> Sur[Surrogate]
        Manager --> DFT[DFT Runner]
        Manager --> Train[Pacemaker]
        Manager --> Inf[Inference]
    end

    TQ --> DFT
    TQ --> Inf
    TQ --> Train
```

## Component Interactions

### 1. Configuration (`mlip_autopipec.config`)
-   **Role**: Source of Truth.
-   **Interaction**: All modules import schemas from here. `WorkflowManager` validates input against `MLIPConfig`.

### 2. Database (`mlip_autopipec.core.database`)
-   **Role**: Persistence.
-   **Implementation**: Wraps `ase.db` (SQLite).
-   **Usage**:
    -   `WorkflowManager` polls for pending structures.
    -   Runners (DFT, Inference) write results back.
    -   Ensures ACID properties via SQLite file locking.

### 3. Orchestration (`mlip_autopipec.orchestration`)
-   **WorkflowManager**: State machine. Decides *what* to do next (e.g., "Run DFT" or "Train Potential").
-   **TaskQueue**: Execution engine. Wraps `dask.distributed`. Handles retries and batch submission.

### 4. Execution Modules
-   **DFT**: Executes Quantum Espresso. Uses `QERunner`.
-   **Inference**: Executes LAMMPS. Uses `LammpsRunner`.
-   **Training**: Executes Pacemaker.

## Error Handling Strategy
-   **Task Level**: `TaskQueue` uses `tenacity` to retry transient submission errors.
-   **Job Level**: Runners (e.g., `LammpsRunner`) catch subprocess errors and return a "Failed" result object instead of crashing the worker.
-   **System Level**: `WorkflowManager` checkpoints state to disk (`checkpoint.json`) to allow resumption after crashes.
