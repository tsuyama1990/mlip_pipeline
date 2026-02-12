# System Architecture: PYACEMAKER

## 1. Summary

**PYACEMAKER** is an advanced, automated system designed to democratise the creation and operation of Machine Learning Interatomic Potentials (MLIP), specifically leveraging the Atomic Cluster Expansion (ACE) formalism via the **Pacemaker** engine. The primary goal is to lower the barrier to entry for materials scientists who require "State-of-the-Art" accuracy in atomistic simulations but lack deep expertise in data science or MLIP construction.

The system addresses critical challenges in the field:
1.  **Extrapolation Risk**: Standard MD simulations often venture into unknown configurations where potentials fail. PYACEMAKER mitigates this via a "Hybrid Potential" approach (ACE + ZBL/LJ) and rigorous uncertainty quantification ($\gamma$-monitoring).
2.  **Data Inefficiency**: Random sampling is wasteful. The system employs **Active Learning** with D-Optimality to select only the most information-rich structures for DFT calculation, reducing computational cost by an order of magnitude.
3.  **Operational Complexity**: Manually coordinating DFT, training, and MD is error-prone. The **Orchestrator** component automates the entire "Explore-Label-Train-Deploy" cycle based on a single YAML configuration.

The architecture is built on a modular, container-ready design where the **Orchestrator** coordinates specialized agents: the **Structure Generator** (Explorer), **Oracle** (DFT/Labeler), **Trainer** (ML Engine), **Dynamics Engine** (MD/kMC Runner), and **Validator** (Quality Gate). This separation of concerns ensures scalability, from local workstations to HPC environments, and supports advanced workflows like Adaptive Kinetic Monte Carlo (aKMC) for extending time scales.

## 2. System Design Objectives

### 2.1. Goals
*   **Zero-Config Workflow**: Enable users to run a full active learning pipeline from a single `config.yaml` without writing Python code.
*   **Data Efficiency**: Achieve target accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with <10% of the DFT calculations required by random sampling.
*   **Physics-Informed Robustness**: Guarantee simulation stability even in extrapolation regions by enforcing physical baselines (Core Repulsion) via Delta Learning.
*   **Time-Scale Extension**: Bridge the gap between nanosecond MD and second-scale diffusion/reaction phenomena using aKMC.

### 2.2. Constraints
*   **Strict Typing**: All internal data structures must be defined using **Pydantic V2** to ensure schema validity and type safety.
*   **Dependency Management**: External tools (LAMMPS, Quantum Espresso, Pacemaker, EON) must be interfaced robustly, handling failures (e.g., SCF non-convergence) gracefully.
*   **Reproducibility**: All random seeds, configuration states, and data versions must be tracked.
*   **Code Quality**: The codebase must adhere to strict linting rules (`ruff`, `mypy`) to prevent technical debt.

### 2.3. Success Criteria
*   **Automated Recovery**: The system must automatically recover from at least 90% of common DFT errors (e.g., mixing beta adjustment).
*   **Stability**: MD simulations must not crash with "Segmentation Fault" due to unphysical atomic overlaps; they should instead trigger a controlled "Halt & Diagnose" loop.
*   **Extensibility**: Adding a new exploration strategy (e.g., genetic algorithm) or a new DFT code (e.g., VASP) should require minimal changes to the core logic.

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture with the **Orchestrator** acting as the central brain.

### 3.1. Component Diagram

```mermaid
graph TD
    subgraph "User Space"
        Config[config.yaml]
        User[User / Researcher]
    end

    subgraph "Core System (Orchestrator)"
        Orch[Orchestrator]
        State[State Manager]
        Logger[Logger]
    end

    subgraph "Modules"
        Gen[Structure Generator]
        Oracle[Oracle (DFT)]
        Train[Trainer (Pacemaker)]
        Dyn[Dynamics Engine]
        Val[Validator]
    end

    subgraph "External Engines"
        LAMMPS[LAMMPS (MD)]
        QE[Quantum Espresso (DFT)]
        PACE[Pacemaker (ML)]
        EON[EON (kMC)]
    end

    User --> Config
    Config --> Orch
    Orch --> State
    Orch --> Gen
    Orch --> Oracle
    Orch --> Train
    Orch --> Dyn
    Orch --> Val

    Gen -- "New Structures" --> Orch
    Oracle -- "Labeled Data" --> Orch
    Train -- "Potential.yace" --> Orch
    Dyn -- "High Uncertainty / Halt" --> Orch

    Dyn --> LAMMPS
    Dyn --> EON
    Oracle --> QE
    Train --> PACE
```

### 3.2. Data Flow (The Active Learning Cycle)

1.  **Exploration**: The `Dynamics Engine` runs MD or kMC using the current potential.
2.  **Detection**: The engine monitors extrapolation grade ($\gamma$). If $\gamma > \text{threshold}$, simulation halts.
3.  **Selection**: The `Orchestrator` extracts the "Halt Structure", and the `Structure Generator` creates local candidates (Normal Mode, Random Displacement).
4.  **Filtering**: The `Trainer` (via `pace_activeset`) selects the most informative candidates (D-Optimality).
5.  **Labeling**: The `Oracle` performs DFT calculations on selected candidates (with Periodic Embedding).
6.  **Refinement**: The `Trainer` updates the potential (Fine-tuning) using the new data.
7.  **Deployment**: The new potential is validated by `Validator` and hot-swapped back into `Dynamics Engine`.

## 4. Design Architecture

The system design enforces strict separation of concerns using a Domain-Driven Design (DDD) approach. Data exchange between modules is strictly typed via Pydantic models.

### 4.1. File Structure

```ascii
project_root/
├── config.yaml                 # User Configuration
├── pyproject.toml              # Dependencies & Linter Config
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py             # Entry Point (CLI)
│       ├── constants.py        # Global Constants
│       ├── core/               # Core Infrastructure
│       │   ├── config_parser.py
│       │   ├── state_manager.py
│       │   └── logger.py
│       ├── domain_models/      # Pydantic Schemas
│       │   ├── config.py       # Configuration Models
│       │   ├── datastructures.py # Atoms, Potential, Calculation
│       │   └── enums.py
│       ├── orchestrator/       # Main Control Logic
│       │   └── workflow.py
│       ├── structure_generator/# Structure Generation
│       │   ├── generator.py
│       │   └── policies.py     # Adaptive Policies
│       ├── oracle/             # DFT Interface
│       │   ├── dft_manager.py
│       │   └── embedding.py    # Periodic Embedding
│       ├── trainer/            # Pacemaker Interface
│       │   ├── pace_wrapper.py
│       │   └── active_set.py
│       ├── dynamics/           # Simulation Engines
│       │   ├── md_driver.py    # LAMMPS Interface
│       │   └── kmc_driver.py   # EON Interface
│       ├── validator/          # Quality Assurance
│       │   ├── physics_check.py
│       │   └── reporting.py
│       └── utils/
│           └── converters.py   # ASE <-> Internal converters
└── tests/                      # Unit & Integration Tests
```

### 4.2. Key Data Models (Pydantic)

*   **`WorkflowState`**: Tracks the current iteration, best potential path, and accumulated dataset statistics.
*   **`Structure`**: A wrapper around atomistic data (positions, cell, numbers) with added metadata (provenance, uncertainty tags).
*   **`CalculationResult`**: Stores Energy, Forces, Stress, and exit codes from DFT.
*   **`PotentialArtifact`**: Represents a trained `.yace` file and its associated metadata (RMSE, training date).

## 5. Implementation Plan (8 Cycles)

The development is divided into 8 sequential cycles to ensure steady progress and testability.

### **CYCLE 01: Core Infrastructure & Domain Models**
*   **Goal**: Establish the foundation of the system.
*   **Features**:
    *   Define Pydantic models for Configuration, Structures, and Workflow State.
    *   Implement robust YAML configuration loading and validation.
    *   Set up centralized logging and state management (save/resume capability).
    *   Create the basic CLI entry point.

### **CYCLE 02: Structure Generator & Initial Exploration**
*   **Goal**: Enable the system to create and explore atomic structures.
*   **Features**:
    *   Implement `StructureGenerator` base class.
    *   Implement `M3GNetGenerator` for "Cold Start" (pre-screening).
    *   Implement `AdaptivePolicy` to dynamically switch between MD, MC, and Defect generation based on material properties.
    *   Implement simple Random Structure Search (RSS).

### **CYCLE 03: Oracle (DFT Automation)**
*   **Goal**: Automate the generation of "Ground Truth" data.
*   **Features**:
    *   Implement `Oracle` interface for DFT codes.
    *   Implement `EspressoDriver` (Quantum Espresso) with strict input validation.
    *   **Self-Healing**: Implement logic to retry failed calculations (e.g., adjust mixing beta).
    *   **Periodic Embedding**: Implement logic to cut out local clusters and embed them in supercells for DFT.

### **CYCLE 04: Trainer (Pacemaker Integration)**
*   **Goal**: Enable the training of ACE potentials.
*   **Features**:
    *   Implement `Trainer` wrapper for `pace_train` CLI.
    *   Implement **Delta Learning** logic (handling LJ/ZBL baselines).
    *   Implement `ActiveSetSelector` wrapping `pace_activeset` (D-Optimality filtering).
    *   Manage dataset accumulation (`.pckl.gzip`).

### **CYCLE 05: Dynamics Engine (LAMMPS MD)**
*   **Goal**: Run MD simulations and detect uncertainty.
*   **Features**:
    *   Implement `MDDriver` for LAMMPS.
    *   **Hybrid Potential**: Automate `pair_style hybrid/overlay` generation (ACE + ZBL).
    *   **Watchdog**: Implement `fix halt` logic based on $\gamma$ (extrapolation grade).
    *   Implement the "Halt & Diagnose" return mechanism.

### **CYCLE 06: Orchestrator & Active Learning Loop**
*   **Goal**: Connect all components into a closed loop.
*   **Features**:
    *   Implement the main `run_cycle` loop in `Orchestrator`.
    *   Logic: Explore -> Halt -> Select (Candidates) -> Label (Oracle) -> Train -> Deploy.
    *   Implement file management (organising `iter_001`, `iter_002`...).
    *   Integration test of the full loop (using Mock components if necessary).

### **CYCLE 07: Advanced Dynamics (kMC / EON)**
*   **Goal**: Extend time scales using Adaptive Kinetic Monte Carlo.
*   **Features**:
    *   Implement `KMCDriver` interfacing with EON software.
    *   Create the Python bridge (`driver.py`) allowing EON to call Pacemaker potentials.
    *   Implement "On-the-Fly" detection within kMC (saddle point search halts).

### **CYCLE 08: Validation & Reporting**
*   **Goal**: Ensure quality and visualize results.
*   **Features**:
    *   Implement `Validator` suite: Phonon stability, Elastic constants, EOS curves.
    *   Implement `ReportGenerator` to create HTML summaries of the training progress (RMSE plots, cumulative uncertainty).
    *   Final System Polish and Documentation.

## 6. Test Strategy

Testing is integral to the development process, enforced by CI pipelines.

### 6.1. Unit Testing
*   **Scope**: Individual functions and classes.
*   **Tool**: `pytest`.
*   **Coverage Target**: > 80%.
*   **Approach**:
    *   Mock external binary calls (LAMMPS, QE, Pacemaker) using `unittest.mock`.
    *   Test Pydantic validation rules (e.g., ensure negative energies raise warnings).
    *   Test file I/O safety (atomic writes).

### 6.2. Integration Testing
*   **Scope**: Interaction between two or more modules.
*   **Approach**:
    *   **Oracle + Trainer**: Ensure atoms labeled by Oracle can be read by Trainer.
    *   **Generator + Dynamics**: Ensure generated structures run in LAMMPS without immediate errors.
    *   **Mock Integration**: Run the full Orchestrator loop using "Mock Engines" (simulated DFT and MD) to verify logic flow without heavy computation.

### 6.3. User Acceptance Testing (UAT)
*   **Scope**: End-to-end user scenarios.
*   **Format**: Jupyter Notebooks.
*   **Key Scenario**: Fe/Pt deposition on MgO.
    *   Phase 1: Separate training of MgO and FePt.
    *   Phase 2: Interface training.
    *   Phase 3: Deposition MD (observing nucleation).
    *   Phase 4: Ordering kMC (observing L10 formation).
