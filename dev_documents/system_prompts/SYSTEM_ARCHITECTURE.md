# System Architecture: PYACEMAKER

## 1. Summary

The **PYACEMAKER** project aims to democratise the creation of Machine Learning Interatomic Potentials (MLIP) by automating the complex workflow of data generation, training, and validation. At its core, it leverages the **Pacemaker** (Atomic Cluster Expansion - ACE) framework to produce state-of-the-art potentials that rival Density Functional Theory (DFT) accuracy at a fraction of the computational cost.

The system addresses the "expert barrier" in materials science, where high-quality potential development typically requires deep knowledge of both quantum mechanics and machine learning. PYACEMAKER introduces a **Zero-Config Workflow**, where a single configuration file orchestrates the entire lifecycle: from initial random structure generation to active learning cycles that iteratively refine the potential against "hard" examples (e.g., high-temperature liquids, defects, interfaces).

The architecture is built around a central **Orchestrator** that manages a suite of decoupled components: a **Structure Generator** for exploring chemical space, an **Oracle** (DFT) for ground-truth labelling, a **Trainer** for potential fitting, and a **Dynamics Engine** for on-the-fly (OTF) exploration and uncertainty quantification. By integrating **Adaptive Kinetic Monte Carlo (aKMC)**, the system also bridges the time-scale gap, allowing the potential to be trained on rare events and long-term diffusion processes, not just picosecond-scale molecular dynamics.

## 2. System Design Objectives

### 2.1. Goals
1.  **Zero-Config Automation**: Enable users to go from a chemical composition (e.g., "FePt") to a production-ready `.yace` potential file without writing Python scripts.
2.  **Data Efficiency**: Achieve target accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) using 1/10th of the DFT calculations compared to random sampling, via Active Learning.
3.  **Physics-Informed Robustness**: Guarantee simulation stability by enforcing physical baselines (Lennard-Jones/ZBL) for core repulsion, preventing "holes" in the potential energy surface.
4.  **Time-Scale Scalability**: Seamlessly integrate MD (short-time) and aKMC (long-time) to capture a comprehensive range of material behaviours.

### 2.2. Constraints
-   **Modularity**: Components must be swappable (e.g., QE vs VASP) and container-friendly.
-   **Recoverability**: The system must handle DFT convergence failures and MD crashes (segmentation faults) gracefully through self-healing mechanisms.
-   **Reproducibility**: All random seeds and configuration parameters must be logged to ensure deterministic results where possible.

### 2.3. Success Criteria
-   **Grand Challenge**: Successfully simulate the hetero-epitaxial growth of FePt nanoparticles on an MgO substrate, observing L10 ordering.
-   **Stability**: No simulation crashes due to non-physical atomic overlaps during the OTF loops.
-   **Accuracy**: Validation metrics (Phonon spectra, Elastic constants) matching DFT within 10%.

## 3. System Architecture

The system follows a hub-and-spoke architecture where the **Orchestrator** coordinates data flow between specialised agents.

```mermaid
graph TD
    subgraph "Core System"
        Orch[Orchestrator]
        Config[Global Config]
        DB[(Dataset / File System)]
    end

    subgraph "Agents"
        Gen[Structure Generator]
        Oracle[Oracle (DFT Manager)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (MD/aKMC)]
        Val[Validator]
    end

    User[User] -->|config.yaml| Config
    Orch -->|Read| Config
    Orch -->|Request Structures| Gen
    Gen -->|Candidate Structures| Orch
    Orch -->|Submit Candidates| Oracle
    Oracle -->|Labelled Data| DB
    DB -->|Training Set| Trainer
    Trainer -->|Potential (.yace)| Orch
    Orch -->|Deploy Potential| Dyn
    Dyn -->|Exploration & Uncertainty| Orch
    Orch -->|Validation Request| Val
    Val -->|Report| User
```

### Data Flow
1.  **Initialisation**: The Orchestrator reads `config.yaml` and initialises the state.
2.  **Seed Generation**: The Structure Generator creates initial random or heuristic structures (M3GNet-screened).
3.  **Labelling**: The Oracle performs DFT calculations on these structures.
4.  **Training**: The Trainer fits an initial ACE potential.
5.  **Active Learning Loop**:
    *   **Exploration**: The Dynamics Engine runs MD/aKMC using the current potential.
    *   **Uncertainty Detection**: High-uncertainty structures ($\gamma > \text{threshold}$) are flagged.
    *   **Selection**: The Orchestrator filters these structures (Active Set) and sends them to the Oracle.
    *   **Refinement**: The potential is retrained with the new data.
6.  **Validation**: The final potential undergoes rigorous physical testing before release.

## 4. Design Architecture

The codebase is structured to enforce separation of concerns, using Pydantic for strict data validation and Abstract Base Classes (ABCs) for component interfaces.

### 4.1. File Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Data Models
│   ├── structure.py        # Atomistic structure definition
│   ├── potential.py        # Potential metadata
│   └── config.py           # Global configuration schema
├── interfaces/             # Abstract Base Classes
│   ├── generator.py
│   ├── oracle.py
│   ├── trainer.py
│   └── dynamics.py
├── core/                   # Main Logic
│   ├── orchestrator.py     # Central control loop
│   └── state.py            # Workflow state management
├── components/             # Concrete Implementations
│   ├── generator/
│   │   ├── random.py
│   │   └── adaptive.py     # Policy-based generation
│   ├── oracle/
│   │   ├── qe.py           # Quantum Espresso handler
│   │   └── vasp.py         # VASP handler (future)
│   ├── trainer/
│   │   └── pacemaker.py    # Pacemaker wrapper
│   ├── dynamics/
│   │   ├── lammps.py       # LAMMPS interface
│   │   └── eon.py          # EON (aKMC) interface
│   └── validator/
│       └── suite.py        # Phonon/Elastic tests
└── main.py                 # CLI Entry point
```

### 4.2. Key Data Models

*   **Structure**: Represents an atomic configuration. Includes positions, cell, atomic numbers, and optional properties (energy, forces, stress). Handles serialisation to/from disk.
*   **Dataset**: A collection of Structures. Supports efficient appending and random access for training batches.
*   **GlobalConfig**: The single source of truth for all parameters, validated at startup to prevent runtime configuration errors.

## 5. Implementation Plan

The project is divided into 6 sequential cycles.

### CYCLE 01: Core Infrastructure & Mocks
**Goal**: Establish the project skeleton, CLI, and orchestrator logic with mock components.
-   **Features**:
    -   Project directory structure and `pyproject.toml`.
    -   Pydantic domain models (initial).
    -   Abstract Base Classes (ABCs) for all components.
    -   Mock implementations of Generator, Oracle, Trainer, Dynamics.
    -   `Orchestrator` logic wired to use mocks.
    -   CLI `run` command.

### CYCLE 02: Structure Generator & Data Management
**Goal**: Implement real structure generation and robust data handling.
-   **Features**:
    -   `AdaptiveGenerator`: Policy-based structure creation (Random, Rattle, Molecular Dynamics seeding).
    -   `Dataset` management: Efficient storage (extxyz/jsonl) and retrieval.
    -   Structure deduplication and canonicalisation.

### CYCLE 03: Oracle (DFT Automation)
**Goal**: Implement the connection to Quantum Espresso for ground-truth data generation.
-   **Features**:
    -   `DFTManager`: Interface to ASE/Espresso.
    -   Automatic input generation (K-spacing, Pseudos from SSSP).
    -   **Self-Healing**: Automatic retry logic for SCF convergence failures (mixing beta, smearing).
    -   **Periodic Embedding**: Logic to cut out clusters and embed them in vacuum/bulk for isolated training.

### CYCLE 04: Trainer (Pacemaker Integration)
**Goal**: Integrate the Pacemaker engine for potential fitting.
-   **Features**:
    -   `PacemakerTrainer`: Wrapper around `pace_train`.
    -   **Delta Learning**: Setup for `V_total = V_ACE + V_LJ/ZBL`.
    -   **Active Set Optimization**: Integration of `pace_activeset` (MaxVol) to select the most informative structures.

### CYCLE 05: Dynamics Engine (LAMMPS & OTF)
**Goal**: Implement the On-the-Fly (OTF) learning loop with MD.
-   **Features**:
    -   `LammpsDynamics`: Interface to LAMMPS via Python.
    -   **Hybrid Potential**: Configuration of `pair_style hybrid/overlay`.
    -   **Uncertainty Watchdog**: Monitoring `gamma` values and triggering `fix halt`.
    -   The OTF Loop: Simulation -> Halt -> Select Candidates -> Oracle -> Retrain.

### CYCLE 06: Validation & Orchestration Finalization
**Goal**: Finalise the system with rigorous validation and aKMC integration.
-   **Features**:
    -   `Validator`: Automated phonon dispersion and elastic constant calculation.
    -   `EonDynamics`: Integration with EON for aKMC (long-time evolution).
    -   **Reporting**: Generation of HTML reports (Parity plots, stability metrics).
    -   Full system integration test (The Grand Challenge).

## 6. Test Strategy

We employ a "Shift-Left" testing strategy, validating components early and often.

### 6.1. Unit Testing
-   **Scope**: Individual classes and functions (e.g., `DFTManager`, `Structure` validation).
-   **Tools**: `pytest`, `pytest-cov`.
-   **Approach**: Mock external binaries (QE, LAMMPS, Pacemaker) to test logic without heavy computation.

### 6.2. Integration Testing
-   **Scope**: Interaction between two components (e.g., Orchestrator -> Trainer).
-   **Tools**: Docker containers or local binaries.
-   **Approach**: Use "Mock Mode" where heavy calculations are simulated but file I/O and state transitions are real.

### 6.3. System/End-to-End Testing (UAT)
-   **Scope**: Full pipeline execution.
-   **Scenario**: The "FePt on MgO" Grand Challenge.
-   **Metrics**:
    -   Does the pipeline complete without crashing?
    -   Is the final potential stable?
    -   Are validation metrics within tolerance?
-   **CI vs Production**:
    -   **CI**: Run on a tiny system (e.g., 4 atoms) with very short MD steps to verify the *plumbing*.
    -   **Production**: Run the full scenario on a workstation/cluster.
