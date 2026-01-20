# MLIP-AutoPipe Package Architecture

This document provides a high-level overview of the `mlip_autopipec` package architecture.

## Overview

The `mlip_autopipec` package is designed to automate the lifecycle of Machine Learning Interatomic Potentials (MLIPs). It is structured into several modular components, each responsible for a specific phase of the workflow.

## Modules

### 1. Core (`mlip_autopipec.core`)
- **Purpose**: Provides essential infrastructure services used throughout the application.
- **Components**:
    - `Config`: Pydantic-based configuration schemas (`DFTConfig`, `GlobalConfig`).
    - `Database`: Wrapper around `ase.db` for persistent storage of atomic structures and calculation results (`DatabaseManager`).
    - `Logging`: Centralized logging configuration.
    - `Workspace`: Manages filesystem directories and file paths.

### 2. DFT (`mlip_autopipec.dft`)
- **Purpose**: Handles First-Principles calculations (Cycle 01 & 02).
- **Components**:
    - `QERunner`: Orchestrates Quantum Espresso `pw.x` calculations.
    - `InputGenerator`: Generates valid input files (`pw.in`).
    - `OutputParser`: Parses output logs (`pw.out`) into structured `DFTResult` objects.
    - `Utils`: Helper functions for K-points, pseudopotentials, and magnetism.

### 3. Generator (`mlip_autopipec.generator`)
- **Purpose**: Generates initial atomic structures (Cycle 03).
- **Components**:
    - `Alloy`: Special Quasirandom Structures (SQS) generation.
    - `Defect`: Creation of vacancies, interstitials, and substitutions.
    - `Molecule`: Molecular geometry generation.

### 4. Surrogate (`mlip_autopipec.surrogate`)
- **Purpose**: Selects candidates for training using active learning (Cycle 04).
- **Components**:
    - `MaceClient`: Interface to pre-trained MACE models for force-based screening.
    - `FPSSampler`: Farthest Point Sampling using SOAP descriptors.

### 5. Training (`mlip_autopipec.training`)
- **Purpose**: Trains the MLIP model (Cycle 05).
- **Components**:
    - `PacemakerWrapper`: Interface to the Pacemaker training code.
    - `ZBLCalculator`: Physics baseline implementation.

### 6. Inference (`mlip_autopipec.inference`)
- **Purpose**: Runs Molecular Dynamics simulations using the trained potential (Cycle 06 & 07).
- **Components**:
    - `LammpsRunner`: Orchestrates LAMMPS simulations.
    - `UncertaintyChecker`: Monitors MLIP uncertainty.

### 7. Orchestration (`mlip_autopipec.orchestration`)
- **Purpose**: Manages the overall workflow and state (Cycle 08).
- **Components**:
    - `WorkflowManager`: State machine controlling the pipeline phases.
    - `TaskQueue`: Distributed task execution (Dask).

## Data Flow

1.  **Configuration**: User input is validated against schemas in `core`.
2.  **Generation**: Structures are created by `generator` and stored in `core.database`.
3.  **DFT**: `dft` module picks up candidates, runs calculations, and stores results (`energy`, `forces`, `stress`).
4.  **Training**: `training` module consumes DFT data to produce a `.yace` potential file.
5.  **Inference**: `inference` module uses the potential to run MD and generate new candidates.
6.  **Loop**: New candidates are fed back into the cycle.
