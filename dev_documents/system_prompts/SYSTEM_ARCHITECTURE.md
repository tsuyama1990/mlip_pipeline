# System Architecture

## 1. Summary

The **PYACEMAKER** project is an automated system designed to construct and operate Machine Learning Interatomic Potentials (MLIP) with minimal human intervention. At its core, it utilizes the "Pacemaker" (Atomic Cluster Expansion - ACE) engine to generate state-of-the-art potentials. The system addresses the high barrier to entry in computational materials science by automating the complex workflow of structure sampling, DFT calculation, potential training, and validation.

The system is built as a Python-based **Orchestrator** that manages a closed-loop Active Learning cycle. It integrates various sub-modules: a **Structure Generator** for exploring chemical and structural spaces, an **Oracle** for robust DFT calculations (using Quantum Espresso), a **Trainer** for ACE potential fitting, a **Dynamics Engine** for running MD/kMC simulations with uncertainty monitoring, and a **Validator** for ensuring physical correctness.

By employing an **Adaptive Exploration Policy**, the system dynamically adjusts its sampling strategy based on the material's characteristics (e.g., metals vs. insulators), significantly improving data efficiency compared to random sampling. The ultimate goal is to achieve a "Zero-Config" workflow where a user provides a single configuration file, and the system autonomously produces a high-fidelity potential capable of handling rare events and long-timescale phenomena.

## 2. System Design Objectives

### Goals
1.  **Democratization of MLIP**: Enable researchers with limited ML/DFT expertise to generate high-quality potentials.
2.  **Automation**: Remove the need for manual intervention in the iterative cycle of training and validation.
3.  **Robustness**: Ensure potentials are physically stable, preventing simulation crashes due to non-physical forces (e.g., core overlap).

### Constraints
-   **Computational Cost**: DFT calculations are expensive; the system must minimize the number of required single-point calculations via active learning.
-   **Compatibility**: Must interface seamlessly with standard tools like LAMMPS, Quantum Espresso, and ASE.
-   **Reproducibility**: All workflows must be deterministic and logged.

### Success Criteria
-   **Zero-Config Workflow**: Complete pipeline execution from a single YAML input.
-   **Data Efficiency**: Achieve target accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with <10% of the DFT cost of random sampling.
-   **Physical Safety**: Zero segmentation faults in MD simulations due to potential divergence, ensured by hybrid potentials (ACE + ZBL/LJ).
-   **Scalability**: Support for transition from local active learning to large-scale MD/kMC simulations.

## 3. System Architecture

The system follows a modular architecture orchestrated by a central controller.

### Components

1.  **Orchestrator**: The brain of the system. It manages the workflow state, transitions between phases (Exploration, Selection, Calculation, Training, Validation), and handles error recovery.
2.  **Structure Generator**: Responsible for "Cold Start" and "Adaptive Exploration". It generates initial structures and proposes new candidates based on an adaptive policy (e.g., varying temperature, introducing defects).
3.  **Dynamics Engine (MD/kMC)**: Runs molecular dynamics (LAMMPS) or kinetic Monte Carlo (EON). It includes an "On-the-Fly" (OTF) monitor that interrupts simulations when the uncertainty metric ($\gamma$) exceeds a threshold.
4.  **Oracle**: Manages DFT calculations. It handles input generation, self-correction (fixing convergence errors), and periodic embedding of cluster structures.
5.  **Trainer**: Wraps the Pacemaker engine. It manages dataset curation, active set selection (D-optimality), and potential fitting (delta-learning with physical baselines).
6.  **Validator**: A suite of physical tests (Phonons, Elasticity, EOS) to verify the potential's quality beyond simple test set metrics.

### Data Flow

```mermaid
graph TD
    User[User Config] --> Orchestrator
    Orchestrator -->|1. Initialize| SG[Structure Generator]
    SG -->|Candidate Structures| Oracle

    subgraph Active Learning Loop
        Oracle -->|DFT Data (Energy/Forces)| Dataset[Dataset & Active Set]
        Dataset --> Trainer
        Trainer -->|Potential (YACE)| DE[Dynamics Engine]
        DE -->|Uncertainty Halt| Selection[Structure Selection]
        Selection -->|Extracted Clusters| Oracle
    end

    Trainer -->|Candidate Potential| Validator
    Validator -->|Pass/Fail| Orchestrator

    Orchestrator -->|Final Pot| Deployment
```

## 4. Design Architecture

The system is designed with a Schema-First approach using Pydantic.

### File Structure

```ascii
src/mlip_autopipec/
├── app.py                      # CLI Entry Point
├── constants.py                # Global Constants
├── domain_models/              # Pydantic Schemas
│   ├── config.py               # Configuration Models
│   ├── structure.py            # Structure & Candidate Models
│   └── workflow.py             # Workflow State Models
├── infrastructure/             # Infrastructure Layer
│   ├── logging.py
│   └── io.py
├── orchestration/              # Core Logic
│   ├── workflow.py             # Workflow Manager
│   └── phases/                 # Phase Implementations
│       ├── exploration.py
│       ├── training.py
│       └── validation.py
├── modules/                    # Component Modules
│   ├── structure_gen/          # Structure Generation
│   ├── oracle/                 # DFT Interface (QE)
│   ├── trainer/                # Pacemaker Wrapper
│   └── validator/              # Validation Suite
└── inference/                  # Dynamics Engines
    ├── lammps.py
    └── eon.py
```

### Key Data Models

-   **`Config`**: Aggregates settings for all modules. Validated at startup.
-   **`WorkflowState`**: Tracks the current cycle, phase, dataset statistics, and halt reasons. Persisted to disk for recoverability.
-   **`Candidate`**: Represents a structure proposed for DFT. Includes metadata like "source" (MD halt, random, etc.) and "priority".
-   **`DFTResult`**: Standardized output from the Oracle, containing energy, forces, stress, and convergence status.

## 5. Implementation Plan

The project is divided into 8 sequential cycles.

### CYCLE 01: Foundation
*   **Goal**: Establish the project skeleton, CLI, logging, and core domain models.
*   **Features**:
    *   Project directory structure.
    *   Pydantic models for `Config` and `WorkflowState`.
    *   CLI command `mlip-auto init` and `run-loop` (stub).
    *   Infrastructure for logging and YAML I/O.
    *   Mock interfaces for external tools.

### CYCLE 02: Basic Exploration (Structure Generation)
*   **Goal**: Implement initial structure generation strategies.
*   **Features**:
    *   `StructureGenerator` module.
    *   Random substitution and cell deformation logic.
    *   Integration with Pymatgen/ASE for symmetry analysis.
    *   "Cold Start" strategy implementation.

### CYCLE 03: DFT Oracle
*   **Goal**: Automate robust DFT calculations.
*   **Features**:
    *   `Oracle` module interfacing with Quantum Espresso.
    *   Self-correction logic for SCF convergence failures.
    *   Standardized input generation (SSSP pseudopotentials).
    *   Handling of `DFTResult` data.

### CYCLE 04: Pacemaker Learner
*   **Goal**: Implement the training pipeline.
*   **Features**:
    *   `Trainer` module wrapping `pace_train`.
    *   Dataset management (creation, merging, splitting).
    *   Delta-learning configuration (ZBL/LJ reference).
    *   Active Set selection integration (`pace_activeset`).

### CYCLE 05: Validation Framework
*   **Goal**: Implement physical validation tests.
*   **Features**:
    *   `Validator` module.
    *   Phonon dispersion calculation (via Phonopy).
    *   Elastic constant calculation.
    *   Equation of State (EOS) fitting.
    *   HTML Report generation.

### CYCLE 06: Active Learning Loop (Dynamics Engine)
*   **Goal**: Close the loop with MD-based active learning.
*   **Features**:
    *   `DynamicsEngine` for LAMMPS.
    *   Hybrid potential setup (PACE + ZBL).
    *   `fix halt` implementation for uncertainty monitoring.
    *   Structure extraction from halted trajectories.

### CYCLE 07: Adaptive Strategy
*   **Goal**: Make the exploration smarter.
*   **Features**:
    *   `AdaptivePolicy` engine.
    *   Logic to decide MD/MC ratios and temperature schedules based on material properties (e.g., band gap, bulk modulus).
    *   Periodic Embedding of extracted clusters.

### CYCLE 08: Expansion
*   **Goal**: Scale up and extend to kMC.
*   **Features**:
    *   `EONWrapper` for Kinetic Monte Carlo.
    *   Integration of long-timescale events into learning.
    *   Full system integration test.
    *   Final documentation and polish.

## 6. Test Strategy

### General Approach
-   **Unit Tests**: Mock external dependencies (ASE, LAMMPS, QE) to test logic in isolation.
-   **Integration Tests**: Test the interaction between two modules (e.g., Trainer -> FileSystem, Oracle -> ASE).
-   **End-to-End Tests**: Run a minimal "dry run" cycle to ensure the full pipeline connects correctly.

### Cycle-Specific Testing
-   **Cycle 01**: Verify CLI entry points and Config validation rules.
-   **Cycle 02**: Verify structure generation validity (symmetry, spacing).
-   **Cycle 03**: Test DFT failure handling using mock error logs.
-   **Cycle 04**: Test dataset file creation and Pacemaker command generation.
-   **Cycle 05**: Validate physical checks against known stable structures (e.g., bulk Si).
-   **Cycle 06**: Test the "Halt -> Extract -> Train" loop logic.
-   **Cycle 07**: Verify Policy decisions given different input feature vectors.
-   **Cycle 08**: Test EON interface and long-running job management.
