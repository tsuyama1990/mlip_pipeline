# PYACEMAKER System Architecture

## 1. Summary

The PYACEMAKER project aims to democratize the creation of high-quality Machine Learning Interatomic Potentials (MLIPs). By leveraging the "Pacemaker" (Atomic Cluster Expansion - ACE) engine, the system automates the traditionally manual and expert-dependent workflow of potential generation. The core innovation lies in its "Zero-Config" philosophy, allowing users with minimal data science expertise to generate state-of-the-art potentials for complex materials.

The system orchestrates a complete active learning loop: generating atomistic structures, calculating their properties using Density Functional Theory (DFT), training the ACE potential, and validating its performance through dynamic simulations. Crucially, it addresses the "Extrapolation Problem" in machine learning potentials by incorporating physical baselines (Lennard-Jones/ZBL) and robust uncertainty quantification. This ensures that simulations do not catastrophically fail when encountering unknown atomic configurations.

The architecture is designed as a modular, containerized application driven by a central Python Orchestrator. It integrates standard community tools like LAMMPS for Molecular Dynamics, Quantum Espresso/VASP for DFT (via ASE), and EON for long-timescale Kinetic Monte Carlo simulations. The result is a robust, self-correcting pipeline that autonomously improves the potential until it meets strict quality criteria, enabling large-scale, high-accuracy material simulations.

## 2. System Design Objectives

### Goals
1.  **Zero-Config Automation**: Enable users to go from a chemical composition to a fully trained, validated potential with a single configuration file.
2.  **Physics-Informed Robustness**: Guarantee simulation stability even in extrapolation regions by hybridizing ACE with physical baselines (Core-Repulsion).
3.  **Data Efficiency**: Achieve high accuracy (Energy RMSE < 1 meV/atom) with minimal DFT calculations through intelligent Active Learning and Periodic Embedding.
4.  **Scalability**: Support seamless transition from local workstation execution to HPC environments, bridging femtosecond MD to second-scale kMC.

### Constraints
-   **Dependency Management**: Must manage complex interactions between external binaries (LAMMPS, QE, Pacemaker) and Python libraries.
-   **Computational Cost**: DFT calculations are expensive; the system must minimize "wasteful" calculations of redundant structures.
-   **Error Handling**: Must automatically recover from common HPC and solver failures (e.g., SCF convergence issues) without user intervention.
-   **Modularity**: Components must be loosely coupled to allow swapping of engines (e.g., changing DFT codes or sampling strategies) without rewriting the core logic.

### Success Criteria
-   **Automation Level**: The pipeline completes a full active learning loop without manual intervention.
-   **Stability**: No "Segmentation Faults" in MD simulations due to atomic overlap (ensured by ZBL/LJ baseline).
-   **Accuracy**: Validation metrics meet the target (Force RMSE < 0.05 eV/Å) on hold-out test sets.
-   **Usability**: New users can run a tutorial scenario (e.g., Fe/Pt on MgO) and obtain scientifically meaningful results.

## 3. System Architecture

The system follows a centralized orchestration pattern. The **Orchestrator** manages the state and data flow between specialized workers.

### Component Overview
1.  **Orchestrator**: The brain of the system. Reads config, manages the active learning loop, and handles data persistence.
2.  **Structure Generator (Explorer)**: Proposes new atomic configurations using adaptive policies (MD, MC, defects).
3.  **Oracle**: Performs ground-truth DFT calculations. Handles self-healing of convergence errors.
4.  **Trainer**: Interfaces with Pacemaker to train the ACE potential, managing the dataset and active set selection.
5.  **Dynamics Engine**: Runs simulations (MD/kMC) using the current potential. Responsible for "On-The-Fly" (OTF) uncertainty detection.
6.  **Validator**: Runs a battery of physical tests (Phonons, Elasticity, EOS) to certify the potential.

### Data Flow Diagram

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Manage| Loop{Active Learning Loop}

    subgraph "Core Modules"
        Explorer[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Val[Validator]
    end

    Loop -->|1. Request Structures| Explorer
    Explorer -->|Candidate Structures| Loop

    Loop -->|2. Request Data| Oracle
    Oracle -->|Labeled Data (E, F, S)| Loop

    Loop -->|3. Train| Trainer
    Trainer -->|Potential (.yace)| Loop

    Loop -->|4. Simulate & Check| Dyn
    Dyn -->|Uncertainty / Halt| Loop

    Loop -->|5. Verify| Val
    Val -->|Pass/Fail| Loop

    Loop -->|Final Output| Result[Production Potential]

    %% Data Stores
    DS_Raw[(Raw Structure Store)]
    DS_Train[(Training Dataset)]
    DS_Pot[(Potential History)]

    Explorer -.-> DS_Raw
    Oracle -.-> DS_Train
    Trainer -.-> DS_Pot
```

## 4. Design Architecture

The system is built on a "Schema-First" design using Pydantic. This ensures strict validation of configuration and data exchange between modules.

### File Structure Strategy
The project uses a clean `src` layout.

```ascii
src/mlip_autopipec/
├── config/                  # Configuration Models
│   ├── config_model.py      # Main Pydantic config
│   └── defaults.py          # Default settings
├── domain_models/           # Domain Entities (Pure Data Classes)
│   ├── structures.py        # Atoms, Candidates
│   ├── dataset.py           # Training Data Abstractions
│   └── validation.py        # Validation Results
├── interfaces/              # Abstract Base Classes (Protocols)
│   ├── core_interfaces.py   # Explorer, Oracle, Trainer, etc.
│   └── event_bus.py         # For loose coupling (future)
├── orchestration/           # Control Logic
│   ├── orchestrator.py      # Main Loop
│   ├── state_manager.py     # Resumability logic
│   └── mocks.py             # Mock implementations for testing
├── services/                # Concrete Implementations of Interfaces
│   ├── external/            # Wrappers for external binaries
│   │   ├── lammps_interface.py
│   │   ├── pacemaker_interface.py
│   │   └── espresso_interface.py
│   ├── structure_gen/       # Generation Logic
│   └── validation/          # Physical Validation Logic
├── utils/                   # Shared Utilities
│   ├── logging.py
│   ├── file_io.py
│   └── periodic_table.py
└── main.py                  # CLI Entrypoint
```

### Key Data Models
1.  **`StructureMetadata`**: Wraps `ase.Atoms` with provenance info (generation method, parent structure ID, uncertainty score).
2.  **`TrainingConfig`**: Defines hyperparameters for Pacemaker (cutoff, orders, loss weights).
3.  **`ValidationResult`**: Captures pass/fail status and detailed metrics for each test (e.g., "Elastic Tensor Error").

## 5. Implementation Plan

The development is divided into 6 sequential cycles.

### CYCLE 01: Skeleton & Basic Loop
**Goal**: Establish the project structure, CLI, and a functional "Mock" loop.
-   Setup project with `uv` and `pyproject.toml`.
-   Define core Pydantic Configuration models.
-   Implement the `Orchestrator` class.
-   Create `MockExplorer`, `MockOracle`, `MockTrainer`, `MockValidator`.
-   Implement the main execution loop that iterates through these mocks.
-   **Deliverable**: A CLI command `mlip-pipeline run config.yaml` that prints the loop progress without running real physics.

### CYCLE 02: Trainer & Baseline Integration
**Goal**: Implement the actual connection to `Pacemaker` and handling of physical baselines.
-   Implement `PacemakerTrainer` service.
-   Implement Logic for "Delta Learning": Setup LJ/ZBL potentials.
-   Create `DatasetManager` to handle `.pckl.gzip` files.
-   **Deliverable**: The system can take an existing dataset and train a real `.yace` potential with LJ baseline.

### CYCLE 03: Oracle & Data Management
**Goal**: Automate DFT calculations with self-healing capabilities.
-   Implement `EspressoOracle` (using ASE).
-   Implement `PeriodicEmbedding` logic (cutting clusters into boxes).
-   Implement Error Handling (Self-Correction) for SCF convergence.
-   **Deliverable**: The system can take a structure, run a robust DFT calculation, and return forces/energies.

### CYCLE 04: Structure Generator (The Explorer)
**Goal**: Implement intelligent sampling strategies.
-   Implement `AdaptiveExplorer` service.
-   Implement strategies: `RandomDisplacement`, `MolecularDynamicsSampling` (simple), `DefectGenerator`.
-   **Deliverable**: The system can generate diverse candidate structures from a seed crystal.

### CYCLE 05: Dynamics Engine - MD & On-The-Fly
**Goal**: The heart of Active Learning. Connect LAMMPS and implement the Uncertainty Watchdog.
-   Implement `LammpsMD` service.
-   Implement "Hybrid Potential" writing logic (overlaying ACE + ZBL).
-   Implement the `Halt & Diagnose` loop for uncertainty detection.
-   **Deliverable**: The system can run MD, stop when uncertainty is high, and trigger the Orchestrator to request new data.

### CYCLE 06: Scale-up, Validation & Integration
**Goal**: Add long-timescale capability (kMC) and rigorous validation.
-   Implement `EonKMC` service.
-   Implement `PhysicsValidator` (Phonons, EOS, Elasticity).
-   Final Integration: Run the full `Fe/Pt on MgO` scenario.
-   **Deliverable**: Full Release Candidate.

## 6. Test Strategy

We employ a "Pyramid" testing strategy.

### 1. Unit Tests (Cycle 01-06)
-   **Scope**: Individual functions and classes.
-   **Tool**: `pytest`.
-   **Focus**:
    -   Pydantic model validation (ensure bad configs fail early).
    -   Logic checks (e.g., does the `Halt` detector trigger correctly on threshold crossing?).
    -   Mocking external calls: Use `unittest.mock` to simulate `subprocess.run` calls to LAMMPS/QE/Pacemaker. We **never** run heavy binaries in unit tests.

### 2. Integration Tests (Cycle 02-06)
-   **Scope**: Interaction between two modules (e.g., Orchestrator <-> Trainer).
-   **Tool**: `pytest` with specific markers.
-   **Focus**:
    -   File I/O: Ensure `.yace` files and `.pckl` files are correctly read/written.
    -   Data integrity: Ensure atoms objects are not corrupted during "Periodic Embedding".

### 3. System Tests (Cycle 06)
-   **Scope**: End-to-End workflow.
-   **Strategy**: "Toy Models".
    -   Run the full pipeline on a very simple system (e.g., Aluminum bulk) with very loose convergence criteria and tiny epochs.
    -   Goal: Ensure the loop completes N cycles without crashing, not to produce physical results.

### 4. User Acceptance Tests (UAT)
-   **Scope**: Real scientific scenarios.
-   **Format**: Jupyter Notebooks (`tutorials/`).
-   **Verification**: The "Fe/Pt on MgO" scenario must run (in Mock mode for CI, Real mode for verification).
