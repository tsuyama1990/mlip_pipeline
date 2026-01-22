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
        Builder --> Sur[SurrogatePipeline]
        Sur --> MACE[MaceWrapper]
        Sur --> FPS[Farthest Point Sampling]
        Sur --> CM[CandidateManager]
        CM --> DB
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
-   **Interaction**: All modules import schemas from here. `WorkflowManager` validates input against `MLIPConfig` (aggregating `TargetSystem`, `DFTConfig`, `GeneratorConfig`, `SurrogateConfig`, etc.).
-   **Strictness**: Validation is enforced via Pydantic with `extra="forbid"`.

### 2. Database (`mlip_autopipec.core.database`)
-   **Role**: Persistence Layer.
-   **Implementation**: Wraps `ase.db` (SQLite).
-   **Usage**:
    -   `WorkflowManager` polls for pending structures.
    -   Runners (DFT, Inference) write results (Forces, Energy, Stress) back.
    -   Ensures ACID properties via SQLite file locking.
    -   **CandidateManager**: Handles business logic for creating new candidates (setting default status, generation tags) to keep `DatabaseManager` pure.

### 3. Orchestration (`mlip_autopipec.orchestration`)
-   **WorkflowManager**: State machine. Decides *what* to do next (e.g., "Run DFT" or "Train Potential").
-   **TaskQueue**: Execution engine. Wraps `dask.distributed`. Handles retries and batch submission to local or HPC resources.

### 4. Generator Module (`mlip_autopipec.generator`)
-   **StructureBuilder**: Facade pattern. Orchestrates the generation pipeline.
    -   **Base Generation**: Creates initial bulk/molecular/SQS structures.
    -   **Distortions**: Applies strain/rattle.
    -   **Defects**: Applies vacancies/interstitials.
-   **SQSStrategy**: Generates chemically disordered supercells for alloys.
-   **DefectStrategy**: Introduces point defects.

### 5. Surrogate Module (`mlip_autopipec.surrogate`)
-   **SurrogatePipeline**: Orchestrates the selection of diverse candidates.
    -   **Pre-screening**: Uses `MaceWrapper` (MACE-MP model) to filter unphysical structures (high forces).
    -   **Sampling**: Uses `FarthestPointSampling` (FPS) on descriptors (SOAP) to select a diverse subset.
    -   **Update**: Marks selected candidates in the database via `DatabaseManager`.

### 6. Execution Modules
-   **DFT**: Executes Quantum Espresso. Uses `QERunner`. Handles error recovery (convergence failure).
-   **Inference**: Executes LAMMPS. Uses `LammpsRunner`. Monitors uncertainty.
-   **Training**: Executes Pacemaker.

## Error Handling Strategy
-   **Task Level**: `TaskQueue` uses `tenacity` to retry transient submission errors.
-   **Job Level**: Runners (e.g., `LammpsRunner`, `QERunner`) catch subprocess errors, parse logs for specific failure codes.
-   **System Level**: `WorkflowManager` checkpoints state to disk.

## Data Flow (Generator -> Surrogate)
1.  `StructureBuilder` creates thousands of raw structures.
2.  `CandidateManager` saves them to DB with `status="pending"`.
3.  `SurrogatePipeline` fetches pending structures.
4.  `MaceWrapper` predicts Energy/Forces. High-force structures are rejected (`status="rejected"`).
5.  `FarthestPointSampling` selects top $N$ diverse structures.
6.  Selected structures are updated to `status="selected"`.
7.  `WorkflowManager` picks up selected structures for DFT.
