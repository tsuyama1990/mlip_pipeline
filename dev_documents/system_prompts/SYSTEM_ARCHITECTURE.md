# System Architecture: PYACEMAKER

## 1. Summary

**PYACEMAKER** is an advanced, automated system designed to democratize the creation and operation of Machine Learning Interatomic Potentials (MLIP), specifically leveraging the Atomic Cluster Expansion (ACE) formalism. In the contemporary landscape of computational materials science, the gap between the accuracy of Density Functional Theory (DFT) and the scalability of Classical Molecular Dynamics (MD) is being bridged by MLIPs. However, the barrier to entry remains prohibitively high for non-specialists. Constructing a robust MLIP typically requires deep expertise in data science, physics, and high-performance computing to manage the iterative loop of structure generation, DFT calculation, potential training, and validation.

PYACEMAKER addresses this critical bottleneck by providing a "Zero-Config" workflow. The system automates the entire lifecycle of an MLIP: from the initial exploration of chemical space to the final deployment of a production-ready potential. By abstracting the complexity of the underlying tools—such as **Pacemaker** for training, **Quantum Espresso/VASP** for ground-truth data generation, **LAMMPS** for molecular dynamics, and **EON** for long-timescale evolution—PYACEMAKER allows researchers to focus on material discovery rather than computational plumbing.

The core philosophy of the system is "Physics-Informed Robustness" and "Data Efficiency". Unlike naive approaches that consume vast amounts of random DFT calculations, PYACEMAKER employs an **Active Learning** strategy. It intelligently explores the potential energy surface, identifying regions of high uncertainty (extrapolation) and selectively performing DFT calculations only on structures that maximize information gain (D-optimality). Furthermore, it enforces physical constraints through a hybrid potential approach, overlaying ACE with robust baselines like Lennard-Jones (LJ) or Ziegler-Biersack-Littmark (ZBL) potentials to prevent unphysical behavior in high-energy regimes.

This architecture is built on a modular, decoupled design orchestrated by a central Python controller. It is designed to be scalable, supporting everything from local workstation prototyping to massive HPC deployments. The ultimate goal is to reduce the time-to-solution for high-quality potentials from months to days, enabling rapid screening of new materials for batteries, alloys, and catalysts.

## 2. System Design Objectives

The design of PYACEMAKER is guided by a set of rigorous objectives and constraints to ensure it meets the needs of both novice users and power users.

### 2.1 Goals
1.  **Zero-Configuration Automation**: The primary goal is to minimize user intervention. A user should be able to define a material system (e.g., "Fe-Pt alloy") and a desired accuracy in a single YAML configuration file, and the system should handle the rest. This includes automatic parameter tuning for DFT (k-points, smearing) and MD (timesteps, temperature schedules).
2.  **Maximized Data Efficiency**: DFT calculations are expensive. The system must achieve "State-of-the-Art" accuracy (Energy RMSE < 1 meV/atom, Force RMSE < 0.05 eV/Å) with the minimum number of DFT calls. This is achieved via Active Learning, where the system only learns what it doesn't know.
3.  **Physical Robustness & Safety**: The system must never crash due to unphysical atomic configurations (e.g., nuclear fusion). It must guarantee stability by enforcing core repulsion and providing a "safe mode" fallback using physics-based baselines (LJ/ZBL) when the ML model extrapolates.
4.  **Seamless Scalability**: The architecture must support a smooth transition from short-time MD simulations to long-timescale phenomena (diffusion, ordering) using Adaptive Kinetic Monte Carlo (aKMC).
5.  **Self-Healing Capabilities**: The system must be resilient to common computational failures. If a DFT calculation fails to converge, the system should automatically diagnose the issue (e.g., charge sloshing) and retry with adjusted parameters (e.g., mixing beta, electronic temperature) without halting the entire workflow.

### 2.2 Constraints
*   **External Dependencies**: The system relies on third-party engines (Quantum Espresso, LAMMPS, Pacemaker). It must manage these dependencies gracefully, abstracting their specific input/output formats.
*   **Computational Resources**: The system must be aware of resource limits. It should optimize job scheduling and avoid submitting jobs that exceed available memory or walltime.
*   **Modularity**: Components must be loosely coupled. Replacing the DFT engine from Quantum Espresso to VASP should require changes only in the Oracle module, not the Orchestrator.

### 2.3 Success Criteria
*   **Automation Level**: A complete cycle (Exploration -> DFT -> Training -> Validation) completes without human intervention.
*   **Accuracy**: Final potentials pass rigorous validation checks, including Phonon stability (no imaginary modes) and Elastic constants agreement with DFT (within 15%).
*   **Robustness**: The system successfully handles "Halt" events (high uncertainty) in MD and recovers by learning the local environment.

## 3. System Architecture

The system follows a centralized **Orchestrator Pattern**, where a main controller directs the flow of data and tasks between specialized, independent components.

### 3.1 Components

*   **Orchestrator**: The "Brain" of the system. It reads the configuration, manages the workflow state, triggers transitions between phases (Exploration, Learning, Validation), and handles global error management.
*   **Structure Generator**: The "Explorer". It uses Adaptive Exploration Policies to generate diverse candidate structures. It decides *how* to explore (MD vs MC, Temperature ramping) based on the material's nature (metal vs insulator).
*   **Oracle**: The "Truth Teller". It wraps DFT codes (QE/VASP). It handles input generation, execution monitoring, error recovery (Self-Healing), and parses outputs (Energy, Forces, Stress) into a standardized format. It also handles "Periodic Embedding" to cut clusters from MD snapshots into DFT-computable cells.
*   **Trainer**: The "Learner". It wraps the **Pacemaker** engine. It manages the training dataset, performs Active Set selection (MaxVol) to filter redundant data, and executes the fitting process using Delta Learning (learning the residual difference from a physical baseline).
*   **Dynamics Engine**: The "Runner". It executes simulations using the trained potential. It supports MD (via LAMMPS) and aKMC (via EON). It includes an **Uncertainty Watchdog** that halts simulations when the extrapolation grade ($\gamma$) exceeds a threshold.
*   **Validator**: The "Quality Gate". It runs physics-based tests (Phonons, EOS, Elasticity) to ensure the potential is not just numerically accurate but physically meaningful.

### 3.2 Data Flow (The Active Learning Loop)

1.  **Initialize**: Load config, setup workspace.
2.  **Explore**: Dynamics Engine or Structure Generator creates structures.
3.  **Detect**: Watchdog detects high uncertainty ($\gamma > \text{limit}$).
4.  **Select**: Select diverse structures (D-optimality) around the high-uncertainty point.
5.  **Compute**: Oracle calculates ground truth (DFT) for selected structures (with periodic embedding).
6.  **Train**: Trainer updates the potential with new data.
7.  **Validate**: Validator checks quality. If pass, deploy; if fail, refine.
8.  **Repeat**: Loop until convergence or max cycles.

### 3.3 Architecture Diagram

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]

    subgraph "Core Modules"
        Orch --> SG[Structure Generator]
        Orch --> Oracle[Oracle (DFT)]
        Orch --> Trainer[Trainer (Pacemaker)]
        Orch --> DE[Dynamics Engine]
        Orch --> Val[Validator]
    end

    subgraph "External Engines"
        Oracle --> QE[Quantum Espresso / VASP]
        Trainer --> PM[Pacemaker]
        DE --> LAMMPS[LAMMPS]
        DE --> EON[EON (aKMC)]
        Val --> Phonopy[Phonopy]
    end

    subgraph "Data Store"
        DB[(Dataset .pckl)]
        Pot[(Potential .yace)]
    end

    SG -- "Candidates" --> Oracle
    DE -- "High Uncertainty\nStructures" --> Oracle
    Oracle -- "Labeled Data\n(E, F, V)" --> DB
    DB --> Trainer
    Trainer -- "New Potential" --> Pot
    Pot --> DE
    Pot --> Val
```

## 4. Design Architecture

The system implementation relies on strict **Pydantic** data models for configuration and internal data exchange, ensuring type safety and clear contracts between modules.

### 4.1 File Structure

The project is structured as a Python package `mlip_autopipec` (internal name) inside `src/`.

```ascii
.
├── pyproject.toml              # Dependencies and Linter Config
├── README.md                   # Project Documentation
├── dev_documents/              # AC-CDD Documents
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── ALL_SPEC.md
│   └── CYCLE{xx}/...
├── src/
│   └── mlip_autopipec/         # Main Package
│       ├── __init__.py
│       ├── main.py             # CLI Entrypoint
│       ├── constants.py        # Global Constants
│       ├── core/               # Core Logic
│       │   ├── orchestrator.py
│       │   ├── config_parser.py
│       │   ├── state_manager.py
│       │   └── logger.py
│       ├── domain_models/      # Pydantic Models
│       │   ├── config.py
│       │   ├── datastructures.py
│       │   └── enums.py
│       ├── generator/          # Structure Generator Module
│       │   ├── interface.py
│       │   └── policies.py
│       ├── oracle/             # Oracle Module
│       │   ├── interface.py
│       │   ├── dft_manager.py
│       │   └── embedding.py
│       ├── trainer/            # Trainer Module
│       │   ├── interface.py
│       │   └── pacemaker_wrapper.py
│       ├── dynamics/           # Dynamics Module
│       │   ├── interface.py
│       │   ├── lammps_driver.py
│       │   └── eon_driver.py
│       └── validator/          # Validator Module
│           ├── interface.py
│           └── physics_tests.py
└── tests/
    ├── uat/                    # User Acceptance Tests
    └── unit/                   # Unit Tests
```

### 4.2 Key Data Models

*   **Config**: A nested Pydantic model mirroring the YAML structure. Validates paths, ranges (e.g., temperatures > 0), and enums (e.g., `DFTCode.QUANTUM_ESPRESSO`).
*   **Structure**: A unified wrapper around `ase.Atoms`, adding metadata like `provenance` (how it was generated), `uncertainty_score`, and `label_status`.
*   **WorkflowState**: Tracks the current cycle, iteration, active potential path, and dataset statistics. Saved atomically to JSON to allow resuming.
*   **EvaluationResult**: Stores validation metrics (RMSE, Phonon status) to decide whether to promote a potential.

## 5. Implementation Plan

The project will be implemented in 8 strictly defined cycles.

### Cycle 01: Core Infrastructure & Mocking
*   **Goal**: Establish the skeleton.
*   **Features**:
    *   Project structure setup.
    *   Pydantic configuration parsing.
    *   Logging and Error Handling.
    *   **Mock** implementations of all 5 core components (Generator, Oracle, Trainer, Dynamics, Validator) to prove the Orchestrator logic.
    *   CLI `mlip-runner` entry point.

### Cycle 02: Structure Generator (Exploration)
*   **Goal**: Intelligent structure proposal.
*   **Features**:
    *   Implement `AdaptiveExplorationPolicy`.
    *   Integrate `M3GNet` (optional/mocked) for initial guesses.
    *   Implement Policies: `Random`, `High-T MD`, `Hybrid MD/MC`.
    *   Generate LAMMPS input scripts for exploration.

### Cycle 03: Oracle (DFT Automation)
*   **Goal**: Reliable ground-truth generation.
*   **Features**:
    *   Implement `DFTManager` using `ase.calculators`.
    *   Implement **Self-Healing** logic (auto-retry with easier params).
    *   Implement **Periodic Embedding**: Logic to cut a cluster from a large system and embed it in a small periodic box for DFT.

### Cycle 04: Trainer (Pacemaker Integration)
*   **Goal**: Connecting the Brain.
*   **Features**:
    *   Implement `PacemakerWrapper`.
    *   Automate `pace_collect` (dataset management).
    *   Implement `Active Set Selection` (MaxVol) integration.
    *   Configure **Delta Learning** (LJ/ZBL baselines) in training configs.

### Cycle 05: Dynamics Engine (MD & Uncertainty)
*   **Goal**: The Active Learning Loop.
*   **Features**:
    *   Implement `LAMMPSDriver` with `pair_style hybrid/overlay`.
    *   Implement **Uncertainty Watchdog**: `fix halt` triggered by high $\gamma$.
    *   Parse LAMMPS logs to detect Halt events.

### Cycle 06: Local Learning Loop
*   **Goal**: Closing the loop.
*   **Features**:
    *   Implement the "Halt & Diagnose" response.
    *   Extract halt structures.
    *   Generate local candidates (perturbations).
    *   Select best candidates (Local D-Optimality).
    *   Feed back to Oracle -> Trainer -> Resume Simulation.

### Cycle 07: Scalability (aKMC & EON)
*   **Goal**: Long-timescale simulation.
*   **Features**:
    *   Integrate **EON** (Adaptive Kinetic Monte Carlo).
    *   Write Python driver for EON to call Pacemaker potentials.
    *   Implement Uncertainty check within the aKMC step (Saddle search).

### Cycle 08: Validation & Production
*   **Goal**: Quality Assurance.
*   **Features**:
    *   Implement `Validator` tests: Phonon Dispersion (Phonopy), Elastic Constants, EOS.
    *   Generate HTML Reports.
    *   Finalize CLI and Documentation.

## 6. Test Strategy

Testing will follow the Pyramid model, emphasizing robust Unit Tests and critical Integration/UATs.

### 6.1 Unit Testing
*   **Framework**: `pytest`
*   **Scope**: Every class and function (e.g., config validation, file parsers, math logic).
*   **Mocks**: Heavy use of `unittest.mock` to simulate external binaries (LAMMPS, QE, Pacemaker) and filesystem operations. We verify that the *correct commands* are constructed, not that the binaries run (that's for Integration).
*   **Coverage target**: > 80%.

### 6.2 Integration Testing
*   **Scope**: Interaction between modules (e.g., Orchestrator calling Generator).
*   **Strategy**: Use the "Mock Components" from Cycle 01 to test the flow. For later cycles, use "Mini-Apps" or simplified inputs (e.g., LJ potential instead of full DFT) to run fast integration tests.

### 6.3 User Acceptance Testing (UAT)
*   **Scope**: End-to-End workflows.
*   **Format**: Jupyter Notebooks acting as "Executable Tutorials".
*   **Scenarios**:
    1.  **Training Demo**: Start from scratch, run a few mock cycles, produce a potential.
    2.  **Deposition Demo**: Use a pre-trained potential to simulate Fe/Pt deposition (MD) and ordering (kMC).
*   **CI/CD**: UAT notebooks will have a `IS_CI_MODE` flag to run on tiny systems (2 atoms, 1 step) in GitHub Actions to verify the code paths without needing heavy compute or licenses.
