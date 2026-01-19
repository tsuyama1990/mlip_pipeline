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
        Manager --> Gen[Generator (Module A)]
        Manager --> Sur[Surrogate (Module B)]
        Manager --> DFT[DFT Runner (Module C)]
        Manager --> Train[Pacemaker (Module D)]
        Manager --> Inf[Inference (Module E)]
    end

    TQ --> DFT
    TQ --> Inf
```

## Module Descriptions

- **Config (`mlip_autopipec.config`)**: Defines strict Pydantic schemas for all inputs.
- **Core (`mlip_autopipec.core`)**: Utilities for Database (ASE-DB) and Logging.
- **Generator (`mlip_autopipec.generator`)**: Creates initial atomic structures (SQS, NMS, Defects).
- **Surrogate (`mlip_autopipec.surrogate`)**: Pre-screens candidates using MACE and Farthest Point Sampling.
- **DFT (`mlip_autopipec.dft`)**: Runs Quantum Espresso calculations with automatic error recovery.
- **Training (`mlip_autopipec.training`)**: Prepares datasets and trains ACE potentials using Pacemaker.
- **Inference (`mlip_autopipec.inference`)**: Runs LAMMPS MD simulations and quantifies uncertainty.
- **Orchestration (`mlip_autopipec.orchestration`)**: Manages the state machine and distributed execution.

## Data Flow

1.  **Exploration**: `Generator` produces structures -> `Surrogate` filters them -> `Database`.
2.  **Labeling**: `WorkflowManager` pulls from DB -> `TaskQueue` runs `DFT` -> Results to DB.
3.  **Learning**: `Trainer` pulls labeled data -> Fits Potential -> Saves artifact.
4.  **Inference**: `Inference` runs MD with new Potential -> Detects Uncertainty -> Extracts structures -> DB.
