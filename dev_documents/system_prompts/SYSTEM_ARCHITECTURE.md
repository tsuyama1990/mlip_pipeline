# System Architecture

## 1. Summary

**PyAceMaker** is an advanced, automated system designed to democratise the construction and operation of Machine Learning Interatomic Potentials (MLIP). At its core, it utilises the "Pacemaker" engine (Atomic Cluster Expansion - ACE) to enable users, even those with limited expertise in material science or data science, to generate "State-of-the-Art" potentials with minimal effort.

In modern computational materials science, MLIPs serve as a crucial bridge between the high accuracy of Density Functional Theory (DFT) and the large-scale capabilities of Classical Molecular Dynamics (MD). However, the traditional workflow for constructing high-quality MLIPs is fraught with challenges. It typically requires deep domain knowledge to manually iterate through structure generation, DFT calculation, training, and verification. Common pitfalls include sampling bias, where "rare events" or high-energy configurations are missed, leading to potentials that fail catastrophically in unknown regions. Additionally, the accumulation of redundant data often wastes precious computational resources without improving accuracy.

PyAceMaker addresses these issues through a **Zero-Config Workflow**. By defining a single configuration file, users can trigger an autonomous pipeline that handles everything from initial structure generation to final model deployment. The system employs **Active Learning** to maximise data efficiency, achieving high accuracy (RMSE Energy < 1 meV/atom) with a fraction of the DFT calculations required by random sampling. Furthermore, it enforces **Physics-Informed Robustness** by incorporating a physical baseline (Lennard-Jones or ZBL potential) to ensure stability even in extrapolation regions.

The system is architected as a modular, container-ready solution orchestrated by a central Python controller. It seamlessly integrates various components—Structure Generator, Oracle, Trainer, Dynamics Engine, and Validator—to create a self-improving loop. This architecture not only simplifies the creation of potentials but also ensures their reliability and scalability, capable of extending from local active learning loops to large-scale simulations involving millions of atoms and long-timescale Kinetic Monte Carlo (kMC) studies.

## 2. System Design Objectives

The design of PyAceMaker is guided by four primary objectives, which also serve as the success criteria for the project.

### 2.1. Dramatic Reduction in Man-Hours (Zero-Config Workflow)
The system aims to eliminate the need for manual intervention and custom scripting.
*   **Goal**: Enable a complete pipeline execution—from initial structure generation to a trained, validated potential—using a single YAML configuration file.
*   **Constraint**: The user interface must be simple enough for non-experts, abstracting away the complexities of the underlying engines (LAMMPS, Quantum Espresso, Pacemaker).

### 2.2. Maximisation of Data Efficiency
We aim to produce high-quality potentials with the minimum amount of expensive DFT data.
*   **Goal**: Achieve target accuracy (Energy RMSE < 1 meV/atom, Force RMSE < 0.05 eV/Å) using less than 1/10th of the data required by random sampling strategies.
*   **Strategy**: Utilise **Active Learning** and **D-Optimality** (via `pace_activeset`) to select only the most information-rich structures for training. The system must automatically identify and discard redundant configurations.

### 2.3. Physics-Informed Robustness
A critical flaw in many MLIPs is unphysical behaviour in extrapolation regions (e.g., lack of repulsion when atoms overlap).
*   **Goal**: Guarantee "physical safety" where the potential never predicts non-physical attractive forces at short distances, preventing simulation crashes (segmentation faults).
*   **Strategy**: Implement **Delta Learning**, where the ML model learns the residual difference between DFT and a physical baseline (Lennard-Jones or ZBL). This ensures that the baseline's repulsive core dominates in the absence of data.

### 2.4. Scalability and Extensibility
The architecture must support a wide range of simulation scales and time domains.
*   **Goal**: Seamlessly transition from small-scale active learning loops to large-scale MD and long-timescale kMC simulations.
*   **Design**: The system must use a modular design where components (like the Oracle or Dynamics Engine) can be swapped or upgraded without affecting the core orchestration logic. It must support containerisation for easy deployment on HPC environments.

## 3. System Architecture

The system follows a modular "Hub-and-Spoke" architecture, with the **Orchestrator** acting as the central brain that coordinates the data flow between specialised components.

```mermaid
graph TD
    User[User] -->|Config (yaml)| Orch[Orchestrator]
    Orch -->|Manage| State[State Manager]

    subgraph "Active Learning Loop"
        Orch -->|1. Explore| Gen[Structure Generator]
        Orch -->|2. Simulate & Halt| Dyn[Dynamics Engine]
        Orch -->|3. Label| Oracle[Oracle (DFT)]
        Orch -->|4. Train| Trainer[Trainer (Pacemaker)]
        Orch -->|5. Verify| Valid[Validator]
    end

    Gen -->|Candidates| Oracle
    Dyn -->|Halt Structures| Gen
    Oracle -->|Refined Data| Trainer
    Trainer -->|Potential (yace)| Dyn
    Trainer -->|Potential (yace)| Valid

    subgraph "External Engines"
        Dyn -.-> LAMMPS
        Dyn -.-> EON[EON (aKMC)]
        Oracle -.-> QE[Quantum Espresso/VASP]
        Trainer -.-> Pace[Pacemaker]
    end
```

### Component Descriptions

1.  **Orchestrator**: The central controller that manages the workflow state, handles errors, and facilitates communication between modules. It implements the logic for the active learning cycle.
2.  **Structure Generator**: Responsible for exploring the chemical and structural space. It uses an **Adaptive Exploration Policy** to decide whether to use MD, MC, or random perturbations based on the material's properties (e.g., metal vs. insulator).
3.  **Oracle**: The "Ground Truth" generator. It runs DFT calculations (e.g., using Quantum Espresso) to label structures with energy, forces, and stress. It includes self-healing logic to handle convergence failures automatically.
4.  **Trainer**: Wrapper around the Pacemaker engine. It manages the training of the ACE potential, including **Active Set Selection** (D-Optimality) to filter training data and **Delta Learning** setup.
5.  **Dynamics Engine**: Runs MD and kMC simulations using the trained potential. It features an **On-the-Fly (OTF)** monitoring system that halts the simulation when the extrapolation grade ($\gamma$) exceeds a safety threshold, triggering a retraining cycle.
6.  **Validator**: A quality assurance gatekeeper. It performs rigorous physical tests (Phonon dispersion, Elastic constants, EOS curves) to ensure the potential is not just numerically accurate but physically meaningful.

## 4. Design Architecture

The project is structured to enforce separation of concerns and type safety using Pydantic.

### File Structure
```
.
├── dev_documents/          # Documentation and Specifications
├── src/
│   └── mlip_autopipec/
│       ├── components/     # Core Business Logic Modules
│       │   ├── dynamics/   # LAMMPS/EON integration
│       │   ├── generator/  # Structure generation strategies
│       │   ├── oracle/     # DFT/ASE interfaces
│       │   ├── trainer/    # Pacemaker wrappers
│       │   └── validator/  # Physical validation suites
│       ├── core/           # Framework Utilities
│       │   ├── orchestrator.py  # Main workflow controller
│       │   ├── state_manager.py # Resume/Checkpoint logic
│       │   └── logger.py        # Centralised logging
│       ├── domain_models/  # Pydantic Data Models (Single Source of Truth)
│       │   ├── config.py   # Configuration schemas
│       │   ├── inputs.py   # Structure and Job definitions
│       │   └── results.py  # Calculation outcomes
│       └── main.py         # CLI Entry Point
├── tests/                  # Pytest suite
├── pyproject.toml          # Dependency and Linter config
└── README.md               # User guide
```

### Domain Models and Data Safety
*   **Pydantic V2**: All data structures (Configs, Inputs, Outputs) are defined as Pydantic models. This ensures strict type checking at runtime and provides clear validation errors.
*   **Immutability**: Core data objects are designed to be immutable where possible to prevent side effects during the workflow.
*   **Strict Typing**: The codebase enforces strict type hints (`mypy --strict`), ensuring that interfaces between modules are clearly defined and adhered to.

## 5. Implementation Plan

The project is divided into 8 sequential cycles to ensure incremental delivery and testing.

*   **CYCLE 01: Core Framework & Infrastructure**
    *   **Goal**: Establish the skeleton of the application.
    *   **Features**: Project structure, Pydantic domain models (`Config`, `Structure`, `Job`), Logging system, State Manager, and Mock Components (to allow logic testing without external binaries).
    *   **Deliverable**: A runnable CLI that executes a "Mock" workflow.

*   **CYCLE 02: Structure Generator (Exploration)**
    *   **Goal**: Implement intelligent structure proposal.
    *   **Features**: Adaptive Exploration Policy engine. Integration with M3GNet for "Cold Start" (initial sampling). Implementation of Random, MD, and MC-based generators.
    *   **Deliverable**: A module capable of generating diverse candidate structures based on material properties.

*   **CYCLE 03: Oracle (DFT Automation)**
    *   **Goal**: Automate the generation of ground truth data.
    *   **Features**: `DFTManager` with ASE integration (Quantum Espresso). robust error handling and self-correction (e.g., adjusting mixing beta on convergence failure). Periodic Embedding logic for cutting out cluster models.
    *   **Deliverable**: A robust interface for running DFT calculations that recovers from common errors.

*   **CYCLE 04: Trainer (Pacemaker Integration)**
    *   **Goal**: Automate the potential training process.
    *   **Features**: `PacemakerWrapper` for CLI interaction. Active Set Selection (D-Optimality) to filter data. Setup for Delta Learning (LJ/ZBL baseline configuration).
    *   **Deliverable**: A module that takes labeled structures and produces a `.yace` potential file.

*   **CYCLE 05: Dynamics Engine (LAMMPS Integration)**
    *   **Goal**: Enable MD simulations with uncertainty quantification.
    *   **Features**: `MDInterface` using `lammps` Python module or binary. Hybrid potential setup (`pair_style hybrid/overlay`). Implementation of the `fix halt` watchdog for uncertainty ($\gamma$) monitoring.
    *   **Deliverable**: A module that runs MD and stops automatically when the potential becomes unreliable.

*   **CYCLE 06: On-the-Fly (OTF) Loop**
    *   **Goal**: Close the loop between Dynamics and Training.
    *   **Features**: Orchestration logic connecting all previous modules. "Halt & Diagnose" workflow: Detect Halt -> Generate Local Candidates -> Label -> Train -> Resume.
    *   **Deliverable**: A fully autonomous Active Learning loop.

*   **CYCLE 07: Advanced Dynamics (Deposition & aKMC)**
    *   **Goal**: Address the "Time-Scale" and complex simulation scenarios.
    *   **Features**: Support for `fix deposit` in LAMMPS (for growth simulations). Integration with **EON** for Adaptive Kinetic Monte Carlo (aKMC) to simulate long-timescale ordering.
    *   **Deliverable**: Capability to run the full "Fe/Pt on MgO" user scenario.

*   **CYCLE 08: Validator & Reporting**
    *   **Goal**: Ensure quality and transparency.
    *   **Features**: Physical validation suite (Phonon dispersion, Elastic constants, EOS curves). HTML Report Generator for visualizing convergence and metrics.
    *   **Deliverable**: A production-ready system with comprehensive reporting.

## 6. Test Strategy

Testing is integral to the development process, following a pyramid approach.

### 6.1. Unit Testing
*   **Scope**: Individual functions and classes (e.g., `Config` validation, file parsers, utility functions).
*   **Tool**: `pytest`.
*   **Approach**: heavily use **Mocking** (via `unittest.mock` or custom Mock classes) to isolate components from external dependencies (LAMMPS, QE, Pacemaker).
*   **Metric**: High code coverage (>80%) for logic-heavy modules.

### 6.2. Integration Testing
*   **Scope**: Interactions between modules (e.g., Orchestrator calling the Trainer, Trainer calling the File System).
*   **Strategy**: Use "Mock Components" (implemented in Cycle 01) to simulate the behavior of external engines. For example, a `MockOracle` that returns random energies allows testing the Orchestrator's data flow without running actual DFT.
*   **CI Compatible**: These tests must run quickly and not require heavy physics codes.

### 6.3. User Acceptance Testing (UAT)
*   **Scope**: End-to-end workflow verification.
*   **Scenarios**: Defined in `FINAL_UAT.md` (e.g., MgO formation, Fe/Pt deposition).
*   **Format**: Jupyter Notebooks acting as executable tutorials.
*   **Modes**:
    *   **Mock Mode (CI)**: Uses tiny systems and mock engines to verify the *workflow logic* finishes without error.
    *   **Real Mode**: Runs the full physics simulation (requires HPC/GPU) to verify *scientific validity*.
