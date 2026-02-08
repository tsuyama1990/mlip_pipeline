# System Architecture: PYACEMAKER

## 1. Summary

The **PYACEMAKER** project is an advanced, automated system for constructing and operating Machine Learning Interatomic Potentials (MLIP) using the **Pacemaker** (Atomic Cluster Expansion) engine. It is designed to democratize access to state-of-the-art computational materials science by removing the steep learning curve associated with manual potential fitting. Traditionally, creating a high-quality MLIP requires deep expertise in density functional theory (DFT), statistical mechanics, and machine learning, often involving tedious manual iteration. PYACEMAKER solves this by providing a "Zero-Config" workflow that autonomously orchestrates the entire lifecycle of a potential: from initial structure generation and active learning to final validation and deployment.

At its core, the system employs an **Active Learning** strategy that dramatically reduces the number of expensive DFT calculations required. Instead of random sampling, it uses an **Adaptive Exploration Policy** to intelligently navigate the chemical and structural space, identifying "high-uncertainty" regions where the potential needs improvement. This is coupled with a **Physics-Informed** approach, where a robust physical baseline (Lennard-Jones or ZBL) is enforced to ensure stability even in extrapolation regimes (e.g., high-energy collisions). The system is built as a modular, container-friendly Python application, orchestrated by a central engine that manages data flow between specialized components: the Structure Generator (Explorer), the Oracle (DFT Calculator), the Trainer (Pacemaker Interface), and the Dynamics Engine (MD/kMC Executor).

The ultimate goal is to enable researchers to define a material system (e.g., "Fe-Pt alloy on MgO substrate") in a single configuration file and receive a fully validated, production-ready potential, along with a complete history of the learning process. The system also bridges the time-scale gap in simulations by integrating Molecular Dynamics (MD) for short-term dynamics with Adaptive Kinetic Monte Carlo (aKMC) for long-term evolution, all driven by the same evolving potential.

## 2. System Design Objectives

The design of PYACEMAKER is guided by the following critical objectives, ensuring it meets the needs of both novice users and expert developers:

### 2.1. Zero-Config Automation
**Goal:** Minimise user intervention.
**Constraint:** The system must be operable via a single YAML configuration file.
**Success Criteria:** A user can start a full training run with `mlip-pipeline run config.yaml` without writing any Python code or shell scripts. The system must handle all internal error recovery (e.g., DFT convergence failures) autonomously.

### 2.2. Data Efficiency & Active Learning
**Goal:** Maximize potential accuracy with minimal DFT cost.
**Constraint:** DFT calculations are the primary bottleneck.
**Success Criteria:** Achieve state-of-the-art accuracy (Energy RMSE < 1 meV/atom, Force RMSE < 0.05 eV/Å) using 1/10th the data of random sampling methods. This is achieved through "Active Set Optimization" (D-optimality) and "Uncertainty Quantification" ($\gamma$ metric).

### 2.3. Physics-Informed Robustness
**Goal:** Prevent unphysical behavior in extrapolation regions.
**Constraint:** Pure ML potentials can behave unpredictably far from training data.
**Success Criteria:** The system must enforce a physical baseline (Core-Repulsion via ZBL/LJ) to prevent atomic overlap (nuclear fusion) during high-temperature MD. Simulations must never crash with segmentation faults due to unphysical forces.

### 2.4. Scalability & Modularity
**Goal:** Support diverse computing environments and future extensions.
**Constraint:** Must run on local workstations and HPC clusters (Slurm/PBS).
**Success Criteria:** Components (Oracle, Trainer, etc.) are loosely coupled via defined interfaces (Abstract Base Classes). The system can be easily extended to support new DFT codes (e.g., VASP, CP2K) or potential engines without rewriting the core orchestrator.

### 2.5. Quality Assurance (QA) as Code
**Goal:** Guarantee the reliability of the generated potentials.
**Constraint:** A low RMSE is not enough; physical properties must be correct.
**Success Criteria:** Every generated potential automatically undergoes a "Validation Suite" (Phonon stability, Elastic constants, EOS curves). Potentials that fail these physical checks are rejected or flagged for refinement, ensuring only physically valid models are deployed.

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture, where a central **Orchestrator** manages the workflow and data exchange between specialized, independent components. This ensures modularity and ease of testing.

### 3.1. Component Overview

1.  **Orchestrator (The Brain):**
    -   **Role:** Manages the active learning loop, handles configuration, and coordinates data flow.
    -   **Responsibility:** It decides "what to do next" (e.g., Explore -> Label -> Train -> Validate). It manages the global state and file system.

2.  **Structure Generator (The Explorer):**
    -   **Role:** Proposes new atomic configurations to explore the potential energy surface.
    -   **Responsibility:** Uses an **Adaptive Exploration Policy** to determine the best sampling strategy (MD, MC, Random, Defect insertion) based on current uncertainty and material properties.

3.  **Oracle (The Sage):**
    -   **Role:** Provides ground-truth labels (Energy, Forces, Stress) using DFT.
    -   **Responsibility:** Runs Quantum Espresso (or VASP). Implements **Self-Correction** (adjusting mixing beta, smearing) to fix convergence errors. Performs **Periodic Embedding** to cut small, calculable clusters from large MD snapshots.

4.  **Trainer (The Learner):**
    -   **Role:** Fits the ACE potential to the accumulated data.
    -   **Responsibility:** Wraps the `pacemaker` engine. Manages **Delta Learning** (fitting only the difference from a physical baseline) and **Active Set Optimization** (selecting the most informative structures).

5.  **Dynamics Engine (The Executor):**
    -   **Role:** Runs simulations to test the potential and explore phase space.
    -   **Responsibility:** Wraps **LAMMPS** for MD and **EON** for aKMC. Monitors **Uncertainty ($\gamma$)** in real-time (On-the-Fly) and halts the simulation if the potential enters an unreliable region, triggering a relearning cycle.

6.  **Validator (The Gatekeeper):**
    -   **Role:** Verifies the physical validity of the potential.
    -   **Responsibility:** Calculates phonons, elastic constants, and melting points. Determines if a potential is "Production Ready".

### 3.2. Data Flow Diagram

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Init| Gen[Structure Generator]
    Orch -->|Loop| Dyn[Dynamics Engine]

    subgraph Active Learning Loop
        Gen -->|Candidate Structures| Oracle[Oracle (DFT)]
        Dyn -->|High Uncertainty Structures| Oracle
        Oracle -->|Labeled Data (E, F, S)| DB[(Dataset)]
        DB -->|Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|potential.yace| Val[Validator]
        Val -- Pass --> Orch
        Val -- Fail --> Gen
    end

    Dyn -->|MD/kMC Trajectory| Results[Simulation Results]
    Trainer -->|Physics Baseline| Dyn
```

## 4. Design Architecture

The system is designed using **Domain-Driven Design (DDD)** principles, with a strong emphasis on **Type Safety** (Pydantic) and **Interface Segregation**.

### 4.1. File Structure

```ascii
src/mlip_autopipec/
├── components/           # Implementation of core components
│   ├── dynamics/         # LAMMPS & EON wrappers
│   ├── generator/        # Structure generation logic
│   ├── oracle/           # DFT (ASE/QE) interface
│   ├── trainer/          # Pacemaker interface
│   └── validator/        # Physics validation suite
├── core/                 # Core logic
│   ├── dataset.py        # Data management (JSONL/Pickle)
│   └── orchestrator.py   # Main workflow engine
├── domain_models/        # Pydantic models (Data Transfer Objects)
│   ├── config.py         # Configuration schema
│   ├── structure.py      # Atom/Structure schema
│   └── potential.py      # Potential metadata schema
├── infrastructure/       # External adapters & Mocks
│   └── mocks.py          # Mock components for testing
├── interfaces/           # Abstract Base Classes (ABCs)
│   ├── dynamics.py
│   ├── generator.py
│   ├── oracle.py
│   ├── trainer.py
│   └── validator.py
├── utils/                # Utilities
│   ├── logging.py
│   └── helpers.py
├── constants.py          # Global constants
├── factory.py            # Component Factory (Dependency Injection)
└── main.py               # CLI Entry point
```

### 4.2. Data Models (Pydantic)

The system relies on strict data validation to prevent runtime errors.

-   **`GlobalConfig`**: The root configuration object, validated against `config.yaml`. It contains sub-configs for each component (e.g., `LammpsConfig`, `QeConfig`).
-   **`Structure`**: A unifying representation of atomic structures, convertible to/from `ase.Atoms`. It enforces constraints (e.g., required arrays like positions, numbers, cell).
-   **`Potential`**: Metadata about a trained potential, including its path, version, and validation metrics.

### 4.3. Key Design Patterns

-   **Strategy Pattern:** Used for the `Generator` (Random vs. M3GNet vs. Mutation) and `Oracle` (QE vs. VASP). The Orchestrator interacts with the interface, not the implementation.
-   **Factory Pattern:** `factory.py` instantiates the correct component implementations based on the configuration string.
-   **Observer Pattern:** The `Dynamics Engine` monitors the simulation state and notifies the Orchestrator (via return values/exceptions) when uncertainty is high.

## 5. Implementation Plan

The project will be executed in **6 distinct cycles**, following the AC-CDD methodology.

### Cycle 01: Core Framework & Mocks
-   **Goal:** Establish the skeleton of the application and the CLI.
-   **Features:**
    -   Setup project structure and Pydantic configuration models.
    -   Define Abstract Base Classes (ABCs) for all components.
    -   Implement `Mock` versions of all components (Generator, Oracle, Trainer, Dynamics).
    -   Implement the `Orchestrator` main loop interacting with mocks.
    -   Basic CLI (`mlip-pipeline run`) and logging.

### Cycle 02: Data Management & Structure Generation
-   **Goal:** Handle atomic data and implement the exploration logic.
-   **Features:**
    -   Implement `Dataset` class for efficient storage (JSONL/Pickle) and retrieval.
    -   Implement real `StructureGenerator` using `pymatgen`/`ase`.
    -   Implement `AdaptivePolicy` engine to decide *how* to sample (MD vs MC ratio).
    -   Integration with external libraries (M3GNet/CHGNet for initial guess).

### Cycle 03: Oracle (DFT Automation)
-   **Goal:** Automate ground-truth generation.
-   **Features:**
    -   Implement `Oracle` using `ase.calculators.espresso`.
    -   Implement "Self-Correction" logic for DFT convergence failures.
    -   Implement "Periodic Embedding" to cut clusters from MD snapshots.
    -   Robust error handling for external DFT processes.

### Cycle 04: Trainer (Pacemaker Integration)
-   **Goal:** Enable ACE potential training.
-   **Features:**
    -   Implement `Trainer` wrapper for `pace_train`.
    -   Implement "Delta Learning" setup (LJ/ZBL baseline configuration).
    -   Implement "Active Set Optimization" (`pace_activeset`) to filter data.
    -   Management of `potential.yace` files and versioning.

### Cycle 05: Dynamics Engine (MD/kMC) & OTF
-   **Goal:** Run simulations and close the active learning loop.
-   **Features:**
    -   Implement `Dynamics` wrapper for `LAMMPS`.
    -   Implement `pair_style hybrid/overlay` logic for physics robustness.
    -   Implement On-the-Fly (OTF) uncertainty monitoring (`fix halt`).
    -   Integration with `EON` for aKMC (Long-time scale evolution).

### Cycle 06: Validator & Full Orchestration
-   **Goal:** Quality assurance and final end-to-end integration.
-   **Features:**
    -   Implement `Validator` suite (Phonon, Elasticity, EOS).
    -   Finalize `Orchestrator` logic to connect all real components.
    -   Implement the "Fe/Pt on MgO" UAT scenario.
    -   End-to-end system testing and documentation polish.

## 6. Test Strategy

Testing is integral to the development process, ensuring reliability and scientific correctness.

### 6.1. Unit Testing
-   **Scope:** Individual functions and classes (e.g., `Dataset.append`, `Config.validate`).
-   **Tools:** `pytest`, `pytest-cov`.
-   **Coverage Goal:** > 80%.
-   **Strategy:** Mock all external dependencies (filesystem, subprocess calls to LAMMPS/QE) to ensure tests are fast and deterministic.

### 6.2. Integration Testing
-   **Scope:** Interaction between pairs of components (e.g., Orchestrator -> Trainer, Trainer -> FileSystem).
-   **Strategy:** Use "Mock Mode" configurations where heavy computations (DFT, MD) are replaced by dummy operations, but the data flow and file handling are real. This verifies that the pipeline "plumbing" works correctly without requiring a supercomputer.

### 6.3. End-to-End (E2E) Testing
-   **Scope:** The full `mlip-pipeline run` command.
-   **Strategy:** Run a "Tiny" scenario (e.g., Al bulk, 2 atoms, 1 cycle) in the CI environment. This proves that the system can initialize, run a cycle, and exit successfully.
-   **Scientific Validation:** The UAT scenarios (Fe/Pt on MgO) serve as the ultimate validation, checking if the physics produced by the system matches expected scientific results.
