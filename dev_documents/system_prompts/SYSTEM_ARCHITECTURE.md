# System Architecture - PyAcemaker

## 1. Summary

**PyAcemaker** is an automated system designed to construct and operate "State-of-the-Art" Machine Learning Interatomic Potentials (MLIP) with minimal human intervention. It leverages the **Pacemaker** (Atomic Cluster Expansion - ACE) engine to democratise high-accuracy atomistic simulations.

Traditionally, constructing high-quality MLIPs requires deep expertise in both data science and computational physics, involving a tedious manual cycle of structure generation, DFT calculation, training, and validation. Common pitfalls include sampling bias (missing rare events), accumulation of redundant data (wasting computation), and high maintenance costs when potentials fail during operation.

PyAcemaker addresses these challenges by implementing a **"Zero-Config"** workflow. It automates the entire pipeline—from initial structure generation to active learning loops—allowing users with limited domain knowledge to generate robust potentials. The system features an **Adaptive Exploration Policy** that intelligently navigates chemical space, an **Oracle** that manages DFT calculations with self-healing capabilities, and a **Dynamics Engine** that continuously monitors uncertainty to prevent unphysical extrapolations. By integrating **Active Learning**, PyAcemaker aims to achieve high accuracy (RMSE Energy < 1 meV/atom) with significantly fewer DFT calculations compared to random sampling.

## 2. System Design Objectives

The design of PyAcemaker is guided by the following core objectives and success metrics:

### 2.1. Zero-Config Workflow
The primary goal is to drastically reduce the engineering hours required to build a potential. Users should only need to provide a single configuration file (`config.yaml`) defining the material system. The system must handle all subsequent steps, including initial structure generation, parameter tuning, and loop management, without requiring custom Python scripting from the user.

### 2.2. Data Efficiency
We aim to maximise the information gain per DFT calculation. By utilising **Active Learning** and **D-Optimality** criteria, the system selects only the most informative structures for training. The target is to achieve production-ready accuracy with **1/10th of the DFT computational cost** compared to standard random sampling approaches.

### 2.3. Physics-Informed Robustness
The system must guarantee physical safety in simulations. This is achieved by:
- **Core-Repulsion:** Enforcing a physical baseline (Lennard-Jones or ZBL) to prevent atomic overlap in high-energy collisions.
- **Delta Learning:** Training the ML model to learn the difference between the DFT energy and the physical baseline, ensuring robust behaviour even in extrapolation regions.

### 2.4. Scalability and Extensibility
The architecture must support a seamless transition from local active learning to large-scale simulations. It is designed with a modular approach using **Docker/Singularity** containers, allowing easy deployment on both local workstations and HPC environments. The system supports extension to different dynamics engines (LAMMPS, EON for kMC) and validation protocols.

## 3. System Architecture

The system follows a modular architecture centred around a Python-based **Orchestrator**.

### 3.1. Components

1.  **Orchestrator (The Brain):**
    - Manages the overall workflow state and transitions between phases (Exploration, Selection, Calculation, Refinement).
    - Handles configuration loading and resource allocation.

2.  **Structure Generator (The Explorer):**
    - Proposes candidate atomic arrangements using an **Adaptive Exploration Policy**.
    - Determines strategies based on material properties (e.g., using different defect densities for insulators vs. metals).

3.  **Oracle (The Sage):**
    - Manages Density Functional Theory (DFT) calculations (e.g., Quantum Espresso).
    - Features **Self-Healing** capabilities to automatically recover from convergence failures by adjusting mixing parameters or smearing.
    - Performs **Periodic Embedding** to extract accurate local forces from large MD snapshots.

4.  **Trainer (The Learner):**
    - Wraps the Pacemaker engine to train ACE potentials.
    - Implements **Delta Learning** (ACE + Baseline) and **Active Set Optimization** to select the most representative structures.

5.  **Dynamics Engine (The Executor):**
    - Runs MD (LAMMPS) and kMC (EON) simulations.
    - Monitors **Uncertainty ($\gamma$)** in real-time. If extrapolation is detected (`fix halt`), it triggers a retraining loop.

6.  **Validator (The Gatekeeper):**
    - rigorous testing of generated potentials against physics-based criteria (Phonon stability, Elastic constants, EOS).
    - Decides whether a potential is ready for deployment or needs further refinement.

### 3.2. Architecture Diagram

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]

    subgraph "Core Loop"
        Orch --> SG[Structure Generator]
        Orch --> DE[Dynamics Engine]

        SG -->|Candidates| DB[(Database)]
        DE -->|High Uncertainty Structures| DB

        DB -->|Selected Structures| Oracle[Oracle (DFT)]
        Oracle -->|Labelled Data| DB

        DB -->|Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|Potential (yace)| Val[Validator]

        Val -- Pass --> DE
        Val -- Fail --> SG
    end

    subgraph "External Engines"
        QE[Quantum Espresso]
        LAMMPS[LAMMPS MD]
        EON[EON kMC]
        PM[Pacemaker Core]
    end

    Oracle -.-> QE
    DE -.-> LAMMPS
    DE -.-> EON
    Trainer -.-> PM
```

## 4. Design Architecture

The system is designed with a strict file structure and Pydantic-based data models to ensure type safety and configuration validation.

### 4.1. File Structure

```text
.
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── app.py                  # CLI Entry point
│       ├── config/                 # Configuration Schemas
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── dft_config.py
│       │   ├── training_config.py
│       │   └── workflow_config.py
│       ├── orchestration/          # State Management
│       │   ├── __init__.py
│       │   ├── manager.py
│       │   └── state.py
│       ├── generator/              # Structure Generation
│       │   ├── __init__.py
│       │   ├── policy.py
│       │   └── defects.py
│       ├── dft/                    # Oracle / DFT Interface
│       │   ├── __init__.py
│       │   ├── runner.py           # QE execution
│       │   └── embedding.py        # Periodic Embedding
│       ├── training/               # Pacemaker Interface
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── dataset.py
│       ├── dynamics/               # MD/kMC Engines
│       │   ├── __init__.py
│       │   ├── lammps.py
│       │   └── eon.py
│       └── validation/             # Physics Validation
│           ├── __init__.py
│           ├── phonon.py
│           ├── elastic.py
│           └── eos.py
├── dev_documents/                  # Documentation
│   └── system_prompts/
├── tests/                          # Test Suite
├── pyproject.toml
└── README.md
```

### 4.2. Data Models (Key Concepts)

-   **`WorkflowConfig`**: Top-level configuration defining active modules, resource limits, and cycle parameters.
-   **`DFTResult`**: Standardised output from the Oracle, containing energy, forces, stress tensor, and status metadata.
-   **`CandidateStructure`**: An atomic structure tagged with its origin (Random, MD-Halt, kMC-Saddle) and uncertainty metrics.
-   **`PotentialState`**: Tracks the lineage of a potential, including its version, parent dataset hash, and validation status.

## 5. Implementation Plan

The development is divided into 6 sequential cycles.

### CYCLE 01: Core Framework & Oracle
**Goal:** Establish the project skeleton, configuration management, and a robust DFT execution engine.
-   **Features:**
    -   Pydantic Configuration Schemas (`dft`, `workflow`).
    -   Orchestrator scaffolding (CLI, logging).
    -   `QERunner`: Quantum Espresso execution with error handling (Self-Healing basics).
    -   Basic `Atom` to `DFTInput` conversion.

### CYCLE 02: Structure Generation & Database Management
**Goal:** Implement the ability to generate initial structures and manage training data.
-   **Features:**
    -   `StructureGenerator`: Random and heuristic-based structure creation.
    -   `DatabaseManager`: Saving/Loading ASE atoms to/from `pckl.gzip` (Pacemaker format).
    -   `TrainerWrapper`: Basic interface to `pace_train` and `pace_collect`.
    -   Integration of `StructureGenerator` into the Orchestrator.

### CYCLE 03: Dynamics Engine & Uncertainty
**Goal:** Enable MD simulations with uncertainty monitoring.
-   **Features:**
    -   `LammpsRunner`: Interface to run LAMMPS.
    -   `HybridPotential`: Logic to generate `pair_style hybrid/overlay` (ACE + ZBL/LJ).
    -   `UncertaintyMonitor`: Implementation of `fix halt` based on `max_gamma`.
    -   Parsing logic for LAMMPS logs to detect halts.

### CYCLE 04: Active Learning Loop
**Goal:** Connect the components into a closed-loop Active Learning system.
-   **Features:**
    -   **The Loop:** Exploration -> Detection -> Selection -> Calculation -> Refinement.
    -   `CandidateSelector`: Logic to select structures for DFT (Active Set / D-Optimality basics).
    -   `PeriodicEmbedding`: Extracting local clusters from halted MD frames for DFT.
    -   Orchestrator state machine to manage cycle transitions automatically.

### CYCLE 05: Validation Framework
**Goal:** Ensure the generated potentials are physically valid, not just numerically accurate.
-   **Features:**
    -   `ValidationRunner`: Driver for validation tasks.
    -   `PhononCheck`: Phonopy integration to detect imaginary frequencies.
    -   `ElasticCheck`: Calculation of elastic constants and Born stability criteria.
    -   `EOSCheck`: Equation of State fitting.
    -   Automated "Pass/Fail" gating logic.

### CYCLE 06: Advanced Dynamics & Final Integration
**Goal:** Extend to kMC and refine the exploration policy for the final release.
-   **Features:**
    -   `EONWrapper`: Interface for Adaptive kMC (Time-scale extension).
    -   `AdaptivePolicy`: Full implementation of dynamic sampling parameters (MD/MC ratio, Temp schedule).
    -   Final System Integration and UAT.
    -   Documentation and Polishing.

## 6. Test Strategy

Testing ensures reliability across the complex interaction of physics engines and Python logic.

-   **Unit Tests:**
    -   Focus on individual methods (e.g., config parsing, file writing, output parsing).
    -   Mock external calls to QE, LAMMPS, and Pacemaker to run fast in CI.
    -   **Target:** >90% code coverage.

-   **Integration Tests:**
    -   Test the interaction between two modules (e.g., Orchestrator calling QERunner).
    -   Use small, fast-calculating systems (e.g., Silicon 2-atom cell) to actually run the external binaries if available, or use realistic mocks.

-   **End-to-End (E2E) / UAT:**
    -   Run a full cycle for a simple system.
    -   Verify that `config.yaml` produces a `potential.yace` and a valid `validation_report`.
    -   **Cycle-specific UAT:** Each cycle has dedicated user acceptance scenarios (e.g., "Verify DFT self-healing works", "Verify MD halts on high uncertainty").

-   **Regression Testing:**
    -   Ensure new physics checks (Validation) do not break existing workflows.
