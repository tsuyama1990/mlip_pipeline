# System Architecture: PYACEMAKER

## 1. Summary

**PYACEMAKER** is a high-efficiency automated system for constructing and operating Machine Learning Interatomic Potentials (MLIP). It is designed to democratise atomic simulations by enabling users with limited expertise in data science or computational physics to generate "State-of-the-Art" potentials with minimal effort.

The core of the system is based on "Pacemaker" (Atomic Cluster Expansion - ACE), a powerful framework for fitting interatomic potentials. PYACEMAKER wraps this engine in an autonomous loop that handles the entire lifecycle of a potential: from initial structure generation and First-Principles (DFT) calculation to model training, validation, and deployment in Molecular Dynamics (MD) or Kinetic Monte Carlo (kMC) simulations.

The system addresses critical challenges in modern computational materials science:
1.  **Complexity**: Eliminates the need for manual "hand-holding" of DFT and fitting processes.
2.  **Data Efficiency**: Uses Active Learning to select only the most informative structures for labelling, reducing expensive DFT calculations by over 90% compared to random sampling.
3.  **Robustness**: Enforces physical laws (like core repulsion) via hybrid potentials to prevent simulation crashes in unknown regions.

PYACEMAKER operates as a set of loosely coupled modules orchestrated by a central Python controller, designed to be deployed on local workstations or HPC environments via Docker/Singularity.

## 2. System Design Objectives

### 2.1 Zero-Config Workflow
The primary goal is to allow a user to define a material system (e.g., "Fe-Pt on MgO") in a single YAML configuration file and launch a fully autonomous pipeline. The system must handle all internal parameter tuning, error recovery, and decision-making without human intervention.

### 2.2 Data Efficiency (Active Learning)
The system must maximise the information gain per DFT calculation. Instead of generating massive random datasets, it employs an **Active Learning** strategy. It monitors the uncertainty of the potential during simulations and only triggers expensive DFT calculations when the simulation encounters a "high-uncertainty" configuration. This "Fail-Fast, Learn-Fast" approach ensures that the potential is trained specifically for the regions of phase space relevant to the user's application.

### 2.3 Physics-Informed Robustness
Machine learning models are mathematical approximators that can behave unpredictably outside their training data (extrapolation). To prevent non-physical behaviour (e.g., atoms overlapping without penalty), the system enforces a **Hybrid Potential** architecture. It combines the ML potential (ACE) with a physics-based baseline (Lennard-Jones or ZBL). This ensures that even in unknown regions, fundamental physical laws (like Pauli exclusion/core repulsion) are respected, preventing simulation crashes.

### 2.4 Scalability and Extensibility
The architecture supports a wide range of scales, from small cluster simulations to large-scale bulk MD. It also bridges the time-scale gap by integrating **Kinetic Monte Carlo (kMC)**, allowing the exploration of slow diffusive processes that are inaccessible to standard MD. The modular design allows for easy addition of new Structure Generators, Oracles, or Validation metrics.

## 3. System Architecture

The system follows a modular "Hub-and-Spoke" architecture, where the **Orchestrator** acts as the central hub managing the data flow between specialised worker modules.

### 3.1 Components

1.  **Orchestrator**: The brain of the system. It reads the configuration, initialises modules, and manages the Active Learning loop. It handles state transitions, error logging, and data persistence.
2.  **Structure Generator**: The "Explorer". It proposes new atomic configurations based on adaptive policies (e.g., temperature ramping, defect introduction) or by perturbing high-uncertainty structures found by the Dynamics Engine.
3.  **Dynamics Engine**: The "Runner". It executes MD (via LAMMPS) or kMC (via EON) simulations using the current potential. It monitors the extrapolation grade ($\gamma$) in real-time and halts execution if the potential enters an unknown region.
4.  **Oracle**: The "Sage". It performs high-fidelity DFT calculations (via Quantum Espresso) to label selected structures with energy, forces, and stress. It includes self-healing logic to handle SCF convergence failures.
5.  **Trainer**: The "Learner". It interfaces with Pacemaker to fit the ACE potential to the accumulated dataset. It manages Delta Learning (fitting the residual against a physical baseline) and Active Set selection.
6.  **Validator**: The "Gatekeeper". It runs rigorous physical tests (Phonon stability, Elastic constants, EOS curves) on the trained potential to ensure it is not just numerically accurate but physically meaningful.

### 3.2 Data Flow (The Active Learning Cycle)

```mermaid
flowchart TD
    Config[/Config.yaml/] --> Orchestrator

    subgraph Cycle [Active Learning Cycle]
        direction TB
        Orchestrator -->|1. Deploy Potential| Dynamics[Dynamics Engine\n(MD / kMC)]
        Dynamics -->|2. Halt & Extract| Generator[Structure Generator]
        Generator -->|3. Candidates| Oracle[Oracle\n(DFT / QE)]
        Oracle -->|4. Labeled Data| Trainer[Trainer\n(Pacemaker)]
        Trainer -->|5. New Potential| Validator[Validator]
        Validator -->|6. Pass/Fail| Orchestrator
    end

    Orchestrator -->|Final Output| Production[Production Potential]

    style Orchestrator fill:#f9f,stroke:#333,stroke-width:2px
    style Cycle fill:#e1f5fe,stroke:#333,stroke-dasharray: 5 5
```

1.  **Exploration**: The Dynamics Engine runs simulations.
2.  **Detection**: If uncertainty ($\gamma$) exceeds a threshold, the simulation halts. The problematic structure is extracted.
3.  **Selection**: The Structure Generator creates local candidates around the problematic structure (to learn gradients/curvature).
4.  **Labelling**: The Oracle computes exact Energy, Forces, and Stress for these candidates.
5.  **Refinement**: The Trainer updates the potential using the new data (Fine-tuning).
6.  **Validation**: The Validator checks if the new potential is physically stable. If passed, the loop repeats with the improved potential.

## 4. Design Architecture

The codebase is structured to ensure separation of concerns, type safety, and ease of testing. It heavily relies on **Pydantic** for data validation and **Abstract Base Classes (ABCs)** for interface definition.

### 4.1 File Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Models (Core Data Structures)
│   ├── config.py           # GlobalConfig, ModuleConfigs
│   ├── structure.py        # Structure, AtomsData
│   └── potential.py        # Potential, ExplorationResult
├── interfaces/             # Abstract Base Classes
│   ├── oracle.py           # BaseOracle
│   ├── trainer.py          # BaseTrainer
│   ├── dynamics.py         # BaseDynamics
│   └── ...
├── infrastructure/         # Concrete Implementations
│   ├── oracle/             # QE/ASE implementations
│   ├── trainer/            # Pacemaker implementations
│   ├── dynamics/           # LAMMPS/EON implementations
│   └── mocks.py            # Mock implementations for Cycle 01/Testing
├── orchestrator/           # Core Logic
│   └── simple_orchestrator.py
├── utils/
│   └── logging.py
└── main.py                 # CLI Entry Point
```

### 4.2 Key Data Models

*   **`GlobalConfig`**: The single source of truth for all settings. Nested models (e.g., `OracleConfig`, `TrainerConfig`) allow modular configuration.
*   **`Structure`**: A strict representation of an atomic structure, including positions, cell, atomic numbers, and optional tags/properties. It ensures data integrity before it reaches the Oracle or Trainer.
*   **`Potential`**: Represents a trained model artifact, including its path, version, and metadata (e.g., training metrics).
*   **`ExplorationResult`**: Captures the outcome of a dynamics run, including whether it halted, the halt structure, and the reason.

## 5. Implementation Plan

The project is divided into 6 sequential cycles, following the AC-CDD methodology.

### CYCLE 01: Foundation & Mocks
*   **Goal**: Establish the project skeleton, core interfaces, and a functional "Mock" loop.
*   **Features**:
    *   Define all ABCs (`BaseOracle`, `BaseTrainer`, etc.).
    *   Implement Pydantic Domain Models.
    *   Implement `MockOracle`, `MockTrainer`, `MockDynamics` that simulate behaviour without external tools.
    *   Basic Logging and Configuration parsing.
*   **Deliverable**: A runnable pipeline that prints "Simulation Halted", "Training...", "Validation Passed" using dummy data.

### CYCLE 02: Core Orchestration & Data Management
*   **Goal**: Implement the real logic of the Active Learning loop and robust data handling.
*   **Features**:
    *   `SimpleOrchestrator`: The state machine managing the loop.
    *   `Dataset` Manager: Efficient storage/retrieval of atomic structures (ASE atoms <-> JSON/Pickle).
    *   `StructureGenerator` (Basic): Logic to generate candidates from a halted structure (Random Displacement, Normal Mode).
*   **Deliverable**: An orchestrator that can manage a growing dataset and drive the components logically.

### CYCLE 03: Oracle & Trainer Integration
*   **Goal**: Connect to real physics engines (Quantum Espresso & Pacemaker).
*   **Features**:
    *   `DFTManager` (Oracle): Interface with ASE/Espresso. Implement "Self-Healing" for convergence errors.
    *   `PacemakerWrapper` (Trainer): Interface with `pace_train`. Implement Active Set selection and Delta Learning configuration.
*   **Deliverable**: A pipeline that can run actual DFT calculations and train real ACE potentials (on a small scale).

### CYCLE 04: Dynamics Engine (MD) & Hybrid Potentials
*   **Goal**: Enable real MD simulations with uncertainty monitoring.
*   **Features**:
    *   `MDInterface` (Dynamics): Interface with LAMMPS.
    *   `Hybrid Potential`: Logic to generate LAMMPS input for `pace + zbl/lj`.
    *   `Fix Halt`: Implement the logic to stop LAMMPS when $\gamma > \text{threshold}$.
*   **Deliverable**: The "Exploration" phase is now real. The system can start a simulation and stop it when it finds a "dangerous" structure.

### CYCLE 05: kMC Integration & Advanced Validation
*   **Goal**: Extend to long time-scales and ensure physical validity.
*   **Features**:
    *   `EONWrapper` (Dynamics): Interface with EON for kMC simulations.
    *   `Validator`: Implement Phonon calculation (Phonopy) and Elastic Constant calculation.
    *   `Gatekeeper Logic`: Automated decision to Pass/Fail a potential based on validation metrics.
*   **Deliverable**: A full-featured system capable of bridging MD and kMC, with quality assurance.

### CYCLE 06: Production Readiness & CLI
*   **Goal**: Polish the user experience and finalise the system.
*   **Features**:
    *   `pyacemaker` CLI: Robust command-line interface with `init`, `run`, `status` commands.
    *   End-to-End Integration Tests: Verify the full "Fe/Pt on MgO" scenario.
    *   Documentation & Tutorials: Complete the user guides.
*   **Deliverable**: A production-ready release.

## 6. Test Strategy

Testing is continuous and multi-layered.

### 6.1 Unit Testing
*   **Scope**: Individual classes and functions (e.g., `DFTManager`, `GlobalConfig`).
*   **Approach**: Use `pytest`. Mock external calls (subprocess, file I/O) to ensure tests are fast and deterministic.
*   **Coverage Target**: > 80%.

### 6.2 Integration Testing
*   **Scope**: Interaction between two modules (e.g., Orchestrator -> Oracle).
*   **Approach**: Use "Mock Mode" configurations. For example, test if the Orchestrator correctly calls the Oracle and handles the returned data, without actually running `pw.x`.
*   **Tools**: `pytest` fixtures, temporary directories.

### 6.3 System/UAT Testing
*   **Scope**: The full pipeline from Config to Final Potential.
*   **Approach**:
    *   **Synthetic Scenarios**: Run the full loop with `Mock` components to verify logic flow.
    *   **Real "Tiny" Scenarios**: Run a real loop on a very small system (e.g., 2 atoms) to verify toolchain integration (ASE -> QE -> Pacemaker -> LAMMPS).
*   **Validation**: Check if the final `potential.yace` exists and if the `validation_report.html` is generated.
