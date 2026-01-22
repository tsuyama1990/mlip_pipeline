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

    subgraph "Module A: Generator"
        Manager --> Builder[StructureBuilder]
        Builder --> SQS[SQS Strategy]
        Builder --> Dist[Distortions (Strain/Rattle)]
        Builder --> Defect[Defect Strategy]
    end

    subgraph "Module B: Surrogate"
        Builder --> Sur[SurrogateExplorer]
        Sur --> MACE[MACE Model]
        Sur --> FPS[Farthest Point Sampling]
    end

    subgraph "Execution Layer (HPC)"
        Manager --> DFT[DFT Runner (Quantum Espresso)]
        Manager --> Train[Pacemaker (Training)]
        Manager --> Inf[Inference (LAMMPS)]
    end

    TQ --> DFT
    TQ --> Inf
    TQ --> Train
```

## Component Interactions

### 1. Configuration (`mlip_autopipec.config`)
-   **Role**: Source of Truth.
-   **Interaction**: All modules import schemas from here. `WorkflowManager` validates input against `MLIPConfig` (aggregating `TargetSystem`, `DFTConfig`, `GeneratorConfig`, etc.).
-   **Strictness**: Validation is enforced via Pydantic with `extra="forbid"`.

### 2. Database (`mlip_autopipec.core.database`)
-   **Role**: Persistence Layer.
-   **Implementation**: Wraps `ase.db` (SQLite).
-   **Usage**:
    -   `WorkflowManager` polls for pending structures.
    -   `StructureBuilder` saves initial candidates.
    -   Runners (DFT, Inference) write results (Forces, Energy, Stress) back.
    -   Ensures ACID properties via SQLite file locking.

### 3. Orchestration (`mlip_autopipec.orchestration`)
-   **WorkflowManager**: State machine. Decides *what* to do next (e.g., "Run DFT" or "Train Potential").
-   **TaskQueue**: Execution engine. Wraps `dask.distributed`. Handles retries and batch submission to local or HPC resources.

### 4. Generator Module (`mlip_autopipec.generator`)
-   **StructureBuilder**: Facade pattern. Orchestrates the generation pipeline.
-   **SQSStrategy**: Generates chemically disordered supercells for alloys using `icet` (if available) or random shuffling.
-   **Distortions**: Applies physical distortions.
    -   **Strain**: Applies affine transformations to the cell to explore the Equation of State.
    -   **Rattle**: Applies Gaussian thermal noise to atomic positions.
-   **Defects**: Introduces point defects (vacancies, interstitials) based on Voronoi analysis or random selection.

### 5. Execution Modules
-   **DFT**: Executes Quantum Espresso. Uses `QERunner`. Handles error recovery (convergence failure).
-   **Inference**: Executes LAMMPS. Uses `LammpsRunner`. Monitors uncertainty.
-   **Training**: Executes Pacemaker.

## Error Handling Strategy
-   **Task Level**: `TaskQueue` uses `tenacity` to retry transient submission errors.
-   **Job Level**: Runners (e.g., `LammpsRunner`, `QERunner`) catch subprocess errors, parse logs for specific failure codes (e.g., SCF convergence), and return a structured "Failed" result object.
-   **System Level**: `WorkflowManager` checkpoints state to disk (`checkpoint.json`) to allow resumption after crashes.

## Data Flow (Generator)
1.  User invokes `mlip-auto generate`.
2.  `StructureBuilder` reads `GeneratorConfig` and `TargetSystem`.
3.  `StructureBuilder` calls `SQSStrategy` to create a base supercell.
4.  `StructureBuilder` loops through strain/rattle steps to create a pool of distorted structures.
5.  `StructureBuilder` applies defects to the pool.
6.  Structures are validated (ASE check) and saved to `DatabaseManager` with metadata (provenance, config_type).
