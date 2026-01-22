# MLIP-AutoPipe Architecture

## System Overview

MLIP-AutoPipe is an automated pipeline for generating Machine Learning Interatomic Potentials (MLIPs). It orchestrates structure generation, active learning (surrogate modeling), and ground-truth validation via Density Functional Theory (DFT).

## Core Components

### 1. Configuration (`mlip_autopipec.config`)
-   **Role**: Centralized configuration management using Pydantic schemas.
-   **Key Schemas**: `MLIPConfig` (root), `DFTConfig`, `GeneratorConfig`, `SystemConfig`.
-   **Principle**: Schema-First Development. All inputs are strictly validated before execution.

### 2. Database Layer (`mlip_autopipec.core.database`)
-   **DatabaseConnector**: Handles connection lifecycle to the underlying SQLite database (`ase.db`).
-   **DatabaseManager**: Provides high-level data access methods (CRUD) and metadata management. It isolates the rest of the system from raw SQL/ASE DB details.

### 3. Generator Module (`mlip_autopipec.generator`)
-   **StructureBuilder**: Facade for generating atomic structures.
-   **Strategies**: `SQSStrategy` (alloys), `DefectStrategy` (point defects).
-   **Transformations**: Strain, Rattle.

### 4. Surrogate Module (`mlip_autopipec.surrogate`)
-   **SurrogatePipeline**: Selects diverse candidates for DFT.
-   **MaceWrapper**: Wraps the MACE foundation model for pre-screening.
-   **Sampling**: Implements Farthest Point Sampling (FPS) on descriptors.

### 5. DFT Factory (`mlip_autopipec.dft`)
-   **QERunner**: Orchestrates Quantum Espresso calculations. It handles the execution loop and retry logic.
-   **RecoveryHandler**: Analyzes failures (e.g., convergence errors) and prescribes recovery strategies (e.g., mixing beta reduction).
-   **InputGenerator**: Generates `pw.in` files with correct physics parameters (k-points, flags).
-   **QEOutputParser**: robustly parses output and validates physical integrity (check for NaNs).

### 6. Orchestration (`mlip_autopipec.orchestration`)
-   **WorkflowManager**: State machine that drives the pipeline phases (Generation -> Selection -> DFT -> Training).
-   **TaskQueue**: Manages parallel execution using Dask.

## Interaction Flow

1.  **User** defines `input.yaml`.
2.  **WorkflowManager** validates config and initializes DB.
3.  **Generator** creates thousands of candidate structures -> DB (pending).
4.  **Surrogate** screens candidates, selects diverse subset -> DB (selected).
5.  **DFT Runner** picks selected structures, runs QE, handles errors -> DB (completed/failed).
6.  **Trainer** (Future) trains the MLIP on completed data.

## Design Principles
-   **TDD**: Tests define behavior before implementation.
-   **Robustness**: Explicit error recovery (DFT ladder, retry loops).
-   **Type Safety**: 100% Type Hint coverage enforced by MyPy.
