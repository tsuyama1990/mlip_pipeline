# System Architecture

## 1. Summary

The **MLIP Auto PiPEC** (Machine Learning Interatomic Potentials - Automated Pipeline for Exascale Computing) is a comprehensive system designed to democratize the creation and operation of state-of-the-art Machine Learning Interatomic Potentials (MLIPs). Built around the **Pacemaker** (Atomic Cluster Expansion) engine, it aims to lower the barrier to entry for material scientists by automating the complex workflow of structure generation, First-Principles (DFT) calculation, potential training, and active learning validation.

The system addresses the critical challenges in modern computational materials science: the "expert gap" required to build MLIPs, the inefficiency of random sampling, and the fragility of potentials in extrapolation regions. By implementing a closed-loop **Active Learning** cycle, the system autonomously explores chemical and structural spaces, detects high-uncertainty configurations, and refines the potential using a robust "Oracle" (DFT interface) and "Trainer" (Pacemaker wrapper).

Key innovations include an **Adaptive Exploration Policy** that intelligently selects sampling strategies (MD vs. MC vs. Defects) based on material properties, and a **Physics-Informed Robustness** mechanism that enforces physical core repulsion (via Delta Learning with LJ/ZBL baselines) to prevent simulation crashes. The architecture is modular, container-ready, and orchestrated by a Python-based core that manages the lifecycle of data and computational jobs from local workstations to HPC environments.

## 2. System Design Objectives

### 2.1. Zero-Config Workflow
*   **Goal**: Enable users to go from a chemical composition to a fully trained, production-ready potential with a single configuration file (`config.yaml`).
*   **Constraint**: The system must infer reasonable defaults for hyperparameters (e.g., DFT cutoffs, MD temperatures) based on the input material system, minimizing the need for user intervention.

### 2.2. Data Efficiency
*   **Goal**: Achieve target accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with significantly fewer DFT calculations compared to random sampling.
*   **Method**: Utilize **Active Learning** with D-Optimality (Active Set) selection to prioritize only the most informative structures for labeling.

### 2.3. Physics-Informed Robustness
*   **Goal**: Ensure simulations never fail due to non-physical forces (e.g., lack of core repulsion) in unexplored regions.
*   **Method**: Implement Delta Learning where the ML model learns the residual difference from a physical baseline (Lennard-Jones or ZBL), guaranteeing physical behavior at short interatomic distances.

### 2.4. Scalability and Extensibility
*   **Goal**: Support a seamless transition from small-scale testing to massive production runs.
*   **Constraint**: The architecture must support distinct execution backends (Local vs. Slurm/PBS) and integrate with external solvers (LAMMPS for MD, EON for kMC) without tight coupling.

## 3. System Architecture

The system follows a hub-and-spoke architecture where the central **Orchestrator** coordinates specialized modules.

```mermaid
graph TD
    User[User / Config] -->|Initializes| Orch[Orchestrator]
    Orch -->|Manages| State[Workflow State & DB]

    subgraph "Cycle 1: Exploration"
        Orch -->|Request| Gen[Structure Generator]
        Gen -->|MD/MC/Defects| Candidates[Candidate Structures]
    end

    subgraph "Cycle 2: Oracle"
        Orch -->|Select & Embed| DFT[DFT Oracle (QE/VASP)]
        DFT -->|Forces & Energy| Dataset[Labeled Dataset]
    end

    subgraph "Cycle 3: Training"
        Orch -->|Train| Trainer[Pacemaker Trainer]
        Dataset --> Trainer
        Trainer -->|Produces| Pot[Potential.yace]
    end

    subgraph "Cycle 4: Inference & AL"
        Orch -->|Deploy| MD[Dynamics Engine (LAMMPS/EON)]
        Pot --> MD
        MD -->|Uncertainty (Gamma)| Watchdog[Watchdog Monitor]
        Watchdog -->|High Uncertainty| Halt[Halt & Recovery]
        Halt -->|New Candidates| Gen
    end

    classDef module fill:#f9f,stroke:#333,stroke-width:2px;
    class Orch,Gen,DFT,Trainer,MD module;
```

### Component Interaction Flow
1.  **Exploration**: The `Structure Generator` creates initial structures or the `Dynamics Engine` explores phase space via MD/kMC.
2.  **Detection**: The `Watchdog` monitors the extrapolation grade ($\gamma$). If $\gamma >$ Threshold, the simulation halts.
3.  **Selection**: High-uncertainty structures are extracted. The system applies **Periodic Embedding** to cut out local clusters into valid supercells.
4.  **Labeling**: The `DFT Oracle` computes exact forces and energies for these new structures, employing auto-recovery for convergence failures.
5.  **Refinement**: The `Trainer` updates the potential using the new data, applying Active Set optimization to keep the model compact.
6.  **Deployment**: The new potential is hot-swapped into the `Dynamics Engine` to resume simulation.

## 4. Design Architecture

The system is designed with strict separation of concerns, utilizing **Pydantic** for robust data validation and **Type Hints** for code clarity.

### 4.1. File Structure

```ascii
mlip_autopipec/
├── app.py                      # CLI Entry Point (Typer)
├── config/                     # Configuration Management
│   ├── models.py               # Aggregated Pydantic Models
│   └── schemas/                # Individual Module Schemas
│       ├── dft.py
│       ├── training.py
│       └── ...
├── data_models/                # Core Domain Objects
│   ├── atoms.py                # ASE Atoms Wrappers/Validators
│   └── workflow.py             # State Management Models
├── generator/                  # Structure Generation
│   ├── builder.py
│   └── policies.py             # Adaptive Exploration Logic
├── dft/                        # Oracle Module
│   ├── runner.py               # Abstract Runner & Implementations
│   ├── qe.py                   # Quantum Espresso Interface
│   └── recovery.py             # Error Handling Strategies
├── training/                   # Training Module
│   ├── pacemaker.py            # Pacemaker Wrapper
│   └── dataset.py              # Data Splitting & Preprocessing
├── inference/                  # Dynamics Module
│   ├── lammps.py               # LAMMPS Runner
│   ├── eon.py                  # EON (kMC) Wrapper
│   └── watchdog.py             # Uncertainty Monitoring
├── orchestration/              # Workflow Management
│   ├── loop.py                 # Active Learning Loop Logic
│   └── database.py             # ASE DB Interface
└── utils/
    ├── logging.py
    └── embedding.py            # Periodic Embedding Logic
```

### 4.2. Core Data Models
*   **`MLIPConfig`**: The root configuration object, validated against strict schemas to prevent runtime errors due to misconfiguration.
*   **`WorkflowState`**: A serializable state object tracking the current cycle, iteration, and status of the pipeline, ensuring resumability.
*   **`CandidateStructure`**: Represents an atomic structure with metadata (origin, uncertainty score, status).

## 5. Implementation Plan

The development is divided into 8 sequential cycles to ensure steady progress and testability.

### **Cycle 01: Core Framework & Configuration**
*   **Goal**: Establish the project skeleton, configuration system, and database interfaces.
*   **Features**: Pydantic schemas for `config.yaml`, logging setup, `DatabaseManager` (ASE db wrapper), and Abstract Base Classes for Runners.

### **Cycle 02: Structure Generation (Explorer)**
*   **Goal**: Implement the engine for creating atomic structures.
*   **Features**: `StructureGenerator` class, random/heuristic sampling strategies, and the framework for the **Adaptive Exploration Policy** (determining MD/MC ratios).

### **Cycle 03: Oracle Interface & Data Prep**
*   **Goal**: Build the interface to First-Principles codes.
*   **Features**: `QERunner` (Quantum Espresso), `InputGenerator` with auto-k-spacing, `RecoveryHandler` for SCF convergence errors, and **Periodic Embedding** utilities for cluster cutout.

### **Cycle 04: Training Orchestration**
*   **Goal**: Automate the Pacemaker training process.
*   **Features**: `PacemakerWrapper`, `DatasetBuilder` (formatting ASE atoms for Pacemaker), Delta Learning configuration (LJ/ZBL), and Active Set selection commands.

### **Cycle 05: Inference Engine (MD)**
*   **Goal**: Enable MD simulations with on-the-fly uncertainty monitoring.
*   **Features**: `LammpsRunner`, `MDInterface`, Hybrid Potential setup (`pair_style hybrid/overlay`), and `Watchdog` logic (`fix halt` integration).

### **Cycle 06: Active Learning Orchestrator**
*   **Goal**: Connect the components into a self-healing loop.
*   **Features**: `WorkflowManager` implementing the "Exploration -> Halt -> Embed -> Train -> Resume" cycle, and state persistence logic.

### **Cycle 07: Advanced Expansion (kMC)**
*   **Goal**: Extend capabilities to long timescales.
*   **Features**: `EONWrapper` for Kinetic Monte Carlo, integration of kMC events into the active learning loop, and advanced defect sampling strategies.

### **Cycle 08: Validation Suite & Production Release**
*   **Goal**: Ensure scientific accuracy and system polish.
*   **Features**: `Validator` suite (Phonon, Elasticity, EOS checks), final CLI (`mlip-auto`) polish, end-to-end integration tests, and documentation.

## 6. Test Strategy

### 6.1. Unit Testing
*   **Framework**: `pytest`
*   **Scope**: Every individual module (Validator, Generator, Runner) must have unit tests.
*   **Mocking**: External binaries (QE, LAMMPS, Pacemaker) will be mocked using `unittest.mock` to test Python logic without requiring heavy computation.

### 6.2. Integration Testing
*   **Scope**: Interaction between modules (e.g., Generator -> Database -> Trainer).
*   **Data**: Use lightweight "dummy" potentials and tiny systems (e.g., 2-atom cells) to verify data flow without long wait times.

### 6.3. User Acceptance Testing (UAT)
*   **Format**: Jupyter Notebooks and CLI dry-runs.
*   **Scenarios**: Defined for each cycle (e.g., "Generate 10 structures from config", "Recover from a failed DFT run").
*   **Criteria**: Successful execution of the workflow step and correct state updates in the database.
