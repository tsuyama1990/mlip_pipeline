# System Architecture

## Overview

MLIP-AutoPipe is a modular, event-driven system designed to automate the generation, calculation, and active learning of machine learning interatomic potentials. It orchestrates a pipeline involving structure generation, surrogate model screening, DFT calculations, potential training, and inference.

## Component Diagram

```mermaid
graph TD
    User[User Config] --> Manager[WorkflowManager]

    subgraph "Orchestration Layer"
        Manager --> PE[PhaseExecutor]
        Manager --> DB[(DatabaseManager)]
        Manager --> TQ[TaskQueue (Dask)]
        Manager --> Dash[Dashboard]
    end

    subgraph "Module A: Generator"
        PE --> Builder[StructureBuilder]
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

    subgraph "Module C: DFT"
        PE --> DFT[QERunner]
        DFT --> QE[Quantum Espresso]
        DFT --> DB
    end

    subgraph "Module D: Training"
        PE --> PM[PacemakerWrapper]
        PM --> Pacemaker[Pacemaker]
        PM --> ConfigGen[TrainConfigGenerator]
    end

    subgraph "Module E: Inference (Active Learning)"
        PE --> Inf[LammpsRunner]
        Inf --> LAMMPS[LAMMPS]
        Inf --> AL[EmbeddingExtractor]
        AL --> DB
    end

    TQ --> DFT
```

## Component Descriptions & Interactions

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
-   **WorkflowManager**: State machine. Maintains the global state (Current Generation, Phase).
    -   **Phases**: Idle -> DFT -> Training -> Inference -> (Loop).
-   **PhaseExecutor**: Decouples logic. `WorkflowManager` delegates the specific execution details of each phase (e.g., "Run DFT Batch", "Train Potential") to this class.
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
-   **DFT (`mlip_autopipec.dft`)**: Executes Quantum Espresso. Uses `QERunner`. Handles error recovery (convergence failure). Writes labeled data to DB.
-   **Training (`mlip_autopipec.training`)**: Executes Pacemaker. `PacemakerWrapper` handles data export (ExtXYZ) and configuration generation.
-   **Inference (`mlip_autopipec.inference`)**: Executes LAMMPS.
    -   `LammpsRunner`: Runs MD using the trained potential. Monitors uncertainty (extrapolation grade).
    -   `EmbeddingExtractor`: If uncertainty threshold is exceeded, extracts local atomic environments (candidates) for the next generation.

## Active Learning Loop
1.  **Exploration**: `StructureBuilder` + `SurrogatePipeline` populate DB with initial candidates.
2.  **DFT**: `QERunner` calculates ground truth labels.
3.  **Training**: `PacemakerWrapper` fits a potential to the labeled data.
4.  **Inference**: `LammpsRunner` runs MD with the new potential.
    -   If **Stable**: Loop finishes (or proceeds to next generation).
    -   If **Unstable** (High Gamma): `EmbeddingExtractor` creates new candidates from uncertain regions -> **Back to DFT**.

## Error Handling Strategy
-   **Task Level**: `TaskQueue` uses `tenacity` to retry transient submission errors.
-   **Job Level**: Runners (e.g., `LammpsRunner`, `QERunner`) catch subprocess errors, parse logs for specific failure codes.
-   **System Level**: `WorkflowManager` checkpoints state to disk (`workflow_state.json`).
