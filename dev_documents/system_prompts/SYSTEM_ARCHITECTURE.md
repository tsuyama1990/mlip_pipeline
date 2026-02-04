# PYACEMAKER System Architecture

## 1. Summary

The **PYACEMAKER** project represents a paradigm shift in the construction and operationalization of Machine Learning Interatomic Potentials (MLIPs). Traditionally, the development of high-fidelity potentials, capable of bridging the accuracy of Density Functional Theory (DFT) with the computational efficiency of classical Molecular Dynamics (MD), has been the exclusive domain of experts in both data science and computational physics. This manual, iterative process—involving structure sampling, DFT calculation, model training, and validation—is fraught with inefficiencies, including sampling bias, the accumulation of redundant data, and the risk of physical extrapolation errors in critical simulation regions.

PYACEMAKER addresses these challenges by delivering a fully automated, **"Zero-Config"** pipeline that democratizes access to State-of-the-Art (SOTA) MLIPs. Built around the **Pacemaker** (Atomic Cluster Expansion) engine, the system autonomously orchestrates the entire lifecycle of a potential: from initial structure generation to active learning loops, and finally to production-ready deployment.

The core philosophy of the system is **"Physics-Informed Active Learning."** Unlike naive data-driven approaches that randomly sample the configuration space, PYACEMAKER employs an **Adaptive Exploration Policy** that intelligently navigates the chemical and structural landscape. It uses physical intuition—such as phase stability, defect formation energies, and elastic strain limits—to guide the search for "high-value" configurations. This ensures that the resulting potential is not only accurate in the equilibrium region but also robust in far-from-equilibrium states, such as transition states, fracture zones, and high-temperature liquids.

A critical innovation of this system is its emphasis on **Robustness via Hybrid Potentials**. Purely mathematical fits, like ACE or Neural Networks, can behave unpredictably in regions where training data is sparse (the "extrapolation regime"). PYACEMAKER enforces physical sanity by coupling the ML potential with a baseline physics model (Lennard-Jones or ZBL). This "Delta Learning" approach ensures that even when the ML model is uncertain, the underlying physics prevents catastrophic failures like nuclear fusion (atomic overlap) in MD simulations.

Furthermore, the system is designed for **Autonomy and Self-Healing**. The "Oracle" module, responsible for generating ground-truth DFT data, is equipped with self-correction logic to handle common convergence failures without human intervention. The "Dynamics Engine" features real-time uncertainty monitoring, capable of pausing a simulation the moment it encounters an unknown configuration, triggering a localized learning loop, and resuming the simulation with an updated, "wiser" potential.

In summary, PYACEMAKER transforms the MLIP workflow from a bespoke, artisanal craft into a robust, industrial-grade process. It empowers material scientists to focus on the *application* of potentials—discovering new alloys, catalysts, and battery materials—rather than the tedious *construction* of them.

## 2. System Design Objectives

The architectural decisions for PYACEMAKER are driven by five primary objectives, serving as the North Star for all implementation details.

### 2.1. Zero-Config Workflow (Minimizing Human Time)
The primary metric for success is the reduction of human intervention. A user should be able to define a material system (e.g., "Ti-O system") and a goal (e.g., "high-temperature stability") in a single YAML configuration file. The system must handle all subsequent steps:
*   **Initialization**: Generating reasonable starting structures (crystals, liquids, defects).
*   **Orchestration**: Managing the complex dependency graph of MD, DFT, and Training tasks.
*   **Error Handling**: Automatically recovering from soft failures (e.g., DFT convergence errors, MD instability).
*   **Termination**: Deciding when the potential has reached "production quality" based on rigorous validation metrics.
The goal is to reduce the "Time-to-Potential" from weeks of expert time to days of machine time.

### 2.2. Data Efficiency (Maximizing Compute ROI)
DFT calculations are computationally expensive. A naive approach of "more data is better" is prohibitively costly. The objective is to achieve SOTA accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with the absolute minimum number of DFT calls.
*   **Active Learning**: The system only calculates DFT for structures where the model is uncertain ($\gamma > \text{threshold}$).
*   **D-Optimality**: Among the uncertain structures, the system selects only the most geometrically diverse set (Active Set) that maximizes the information gain (determinant of the feature matrix).
This approach aims to reduce the required training set size by an order of magnitude compared to random sampling.

### 2.3. Physics-Informed Robustness (Safety First)
A potential must never crash a simulation. The "black box" nature of ML models makes them prone to unphysical behavior (e.g., artificial energy wells) in extrapolation regions.
*   **Core Repulsion**: We strictly enforce physical repulsion at short distances using ZBL/LJ baselines.
*   **Delta Learning**: The ML model only learns the *correction* to the baseline physics, not the total energy from scratch.
*   **Hybrid Inference**: In production MD, the potential is always evaluated as $E_{total} = E_{baseline} + E_{ACE}$.
This guarantees that simulations remain stable even under extreme conditions (shock waves, radiation damage) where the ML model might otherwise fail.

### 2.4. Scalability and Flexibility
The architecture must support a wide range of deployment scenarios and material systems.
*   **Containerization**: Every component runs in a reproducible environment (Docker/Singularity).
*   **HPC Integration**: The system abstracts job submission, allowing it to run on a laptop (for testing) or scale to thousands of cores on a supercomputer.
*   **Multi-Scale**: The workflow seamlessly bridges the gap between atomic vibrations (MD) and diffusive events (kMC), enabling the study of phenomena across timescales.

### 2.5. Rigorous Quality Assurance (The "QA Gate")
A potential is useless if it cannot be trusted. The system includes a "Validator" module that acts as a strict gatekeeper.
*   It performs not just statistical validation (RMSE on test sets) but **Physical Validation**: calculating phonon dispersion curves (to check for dynamic stability), elastic constants (mechanical stability), and equations of state.
*   A potential is only marked as "Production Ready" if it passes all these physical checks, preventing the deployment of numerically accurate but physically nonsensical models.

## 3. System Architecture

The system is architected as a modular, loop-based orchestrator where distinct agents (components) collaborate to improve the shared knowledge base (the Potential).

### 3.1. Core Components

1.  **Orchestrator (`The Brain`)**:
    *   The central controller that manages the lifecycle of the active learning loop.
    *   It maintains the state of the system, transitions between phases (Exploration -> Learning), and handles global configuration.

2.  **Structure Generator (`The Explorer`)**:
    *   Responsible for proposing new atomic configurations.
    *   It uses an **Adaptive Exploration Policy** to decide *how* to explore: should it heat the system? Compress it? Introduce defects?
    *   It generates candidate structures that push the boundaries of the current potential's validity.

3.  **Dynamics Engine (`The User / The Tester`)**:
    *   Runs actual simulations (MD via LAMMPS, kMC via EON) using the current potential.
    *   Equipped with an **Uncertainty Watchdog** that monitors the extrapolation grade ($\gamma$) in real-time.
    *   If uncertainty spikes, it halts the simulation and requests "reinforcements" (new training data).

4.  **Oracle (`The Teacher`)**:
    *   The source of ground truth. It wraps DFT codes (Quantum Espresso / VASP).
    *   It receives candidate structures, performs **Periodic Embedding** to ensure valid boundary conditions, and returns precise Energy, Forces, and Stress tensors.
    *   Includes **Self-Healing** logic to retry calculations with adjusted parameters upon failure.

5.  **Trainer (`The Student`)**:
    *   Wraps the Pacemaker engine.
    *   Manages the dataset, performs **Active Set Selection** (D-optimality), and updates the ACE potential weights.
    *   Ensures the potential fits the *residual* between the DFT data and the physics baseline.

6.  **Validator (`The Auditor`)**:
    *   Runs a battery of physical tests (Phonons, Elasticity, EOS) on the newly trained potential.
    *   Provides the "Go/No-Go" decision for deployment.

### 3.2. Data Flow Diagram

```mermaid
graph TD
    subgraph "Control Plane"
        Orch[Orchestrator]
        Config[Configuration]
    end

    subgraph "Exploration & Detection"
        SG[Structure Generator]
        DE[Dynamics Engine]
        WD[Watchdog]
    end

    subgraph "Ground Truth Generation"
        Oracle[Oracle (DFT)]
        Embed[Periodic Embedding]
    end

    subgraph "Learning & Improvement"
        Trainer[Trainer (Pacemaker)]
        AS[Active Set Selector]
        DB[(Structure Database)]
    end

    subgraph "Quality Assurance"
        Val[Validator]
        Rep[Report Generator]
    end

    Config --> Orch
    Orch --> SG
    Orch --> DE

    SG -- "Candidate Structures" --> DE
    DE -- "Running MD/kMC" --> WD
    WD -- "High Uncertainty Detected" --> Embed

    Embed -- "Embedded Clusters" --> Oracle
    Oracle -- "E, F, S Data" --> DB

    DB --> AS
    AS -- "Optimal Training Set" --> Trainer

    Trainer -- "New Potential (YACE)" --> Val
    Val -- "Physical Checks" --> Rep

    Rep -- "Pass/Fail" --> Orch
    Orch -- "Deploy / Retrain" --> DE
```

## 4. Design Architecture

The system follows a **Schema-First** design philosophy. All data flowing between components is strictly defined by Pydantic models, ensuring type safety and clear contracts.

### 4.1. File Structure

```ascii
src/mlip_autopipec/
├── config/                  # Configuration Schemas
│   ├── config_model.py      # Main Pydantic Config Models
│   └── defaults.py          # Default parameter sets
├── domain_models/           # Core Domain Objects
│   ├── structures.py        # Structure, Snapshot, Trajectory
│   ├── potential.py         # Potential metadata, versioning
│   └── validation.py        # Validation results, Metrics
├── interfaces/              # Abstract Base Classes (Protocols)
│   ├── core_interfaces.py   # Explorer, Oracle, Trainer, Validator protocols
│   └── observers.py         # Event listeners
├── orchestration/           # The Brain
│   ├── orchestrator.py      # Main loop logic
│   ├── state_manager.py     # Resume/Checkpoint handling
│   └── mocks.py             # Mock implementations for Cycle 01
├── structure_generation/    # The Explorer
│   ├── generator.py         # Main entry point
│   └── policies.py          # Adaptive exploration policies
├── oracle/                  # The Teacher
│   ├── dft_manager.py       # DFT execution logic
│   └── embedding.py         # Cluster embedding logic
├── training/                # The Student
│   ├── pacemaker_wrapper.py # CLI wrapper for Pacemaker
│   └── active_set.py        # D-Optimality logic
├── dynamics/                # The User
│   ├── lammps_driver.py     # LAMMPS control
│   └── eon_driver.py        # kMC control
├── validation/              # The Auditor
│   ├── stability.py         # Phonon/Elastic checks
│   └── performance.py       # RMSE/Error metrics
└── utils/
    ├── logging.py           # Structured logging
    └── serialization.py     # Data IO
```

### 4.2. Key Data Models

*   **`StructureMetadata`**: Carries provenance information. Where did this structure come from? (Random generation? MD Snapshot? Active Learning selection?). Includes tags for `uncertainty_score` and `parent_structure_id`.
*   **`TrainingConfig`**: Defines the hyperparameters for Pacemaker, including cutoff radii, polynomial degrees, and the choice of baseline potential (LJ/ZBL).
*   **`ExplorationState`**: Tracks the current coverage of the chemical space. Used by the Adaptive Policy to decide the next move (e.g., "We have enough liquid data, let's explore high-pressure crystals").

## 5. Implementation Plan

The project is strictly divided into 6 implementation cycles.

### CYCLE 01: Core Skeleton & Mock Orchestration
**Objective**: Establish the software backbone. The Orchestrator loop must run from start to finish using "Mock" components that simulate work without running heavy computations.
*   **Features**:
    *   Define all Core Interfaces (Protocols) for Explorer, Oracle, Trainer, etc.
    *   Implement the `Orchestrator` main loop.
    *   Create `MockExplorer` (returns random atoms), `MockOracle` (returns LJ forces), `MockTrainer` (touches a dummy file).
    *   Setup Pydantic configuration loading and Logging.
*   **Outcome**: `python main.py` runs a full "virtual" active learning loop, generating logs and dummy artifacts.

### CYCLE 02: Intelligent Structure Generation
**Objective**: Replace `MockExplorer` with the real `StructureGenerator`.
*   **Features**:
    *   Implement `AdaptiveExplorationPolicy` engine.
    *   Integrate `StructureGenerator` with simple random transformations (rattle, strain).
    *   Implement "Material DNA" extraction (basic properties of input structure).
*   **Outcome**: The system can intelligently propose diverse atomic configurations based on policy rules, even if it can't calculate them accurately yet.

### CYCLE 03: The Robust Oracle (DFT)
**Objective**: Replace `MockOracle` with a production-ready DFT pipeline.
*   **Features**:
    *   Implement `DFTManager` using ASE (Quantum Espresso / VASP support).
    *   Implement `PeriodicEmbedding` to carve out clusters from MD snapshots.
    *   Implement Self-Correction logic (auto-fix convergence errors).
*   **Outcome**: The system can reliably turn an atomic structure into a set of ground-truth Forces and Energies, handling calculation failures autonomously.

### CYCLE 04: Training & Active Learning
**Objective**: Replace `MockTrainer` with the real Pacemaker integration.
*   **Features**:
    *   Implement `PacemakerWrapper` to drive `pace_train`.
    *   Implement `ActiveSetSelector` using MaxVol/D-optimality to filter redundant data.
    *   Implement Data Management (merging datasets, managing `.pckl.gzip` files).
*   **Outcome**: The system can actually learn. It takes structure+energy data and produces a valid `.yace` potential file.

### CYCLE 05: Dynamics Engine & OTF Loop
**Objective**: Implement the feedback loop. The potential generated in Cycle 04 is used to drive simulations.
*   **Features**:
    *   Implement `MDInterface` for LAMMPS.
    *   Implement the **Hybrid Potential** setup (ACE + ZBL/LJ).
    *   Implement the **Uncertainty Watchdog** (`fix halt` in LAMMPS) and the "Halt & Diagnose" logic in Python.
*   **Outcome**: The "Active Learning" closes. The system runs MD, detects failure (high uncertainty), and automatically triggers the Orchestrator to request new data.

### CYCLE 06: Integration, kMC & Validation
**Objective**: The final polish. Connect the time-scale bridge (kMC) and the quality gate (Validator).
*   **Features**:
    *   Implement `EONWrapper` for Adaptive kMC.
    *   Implement `Validator` (Phonon, Elastic, EOS checks).
    *   Finalize the `Orchestrator` to handle the full complex workflow (MD -> Halt -> Train -> kMC -> Validate).
    *   Generate Tutorials (Fe/Pt on MgO).
*   **Outcome**: A fully functional, autonomous system capable of solving the "Fe/Pt on MgO" grand challenge.

## 6. Test Strategy

### General Philosophy
We employ a **"Pyramid"** testing strategy.
*   **Unit Tests (70%)**: Test individual functions and classes in isolation using Mocks.
*   **Integration Tests (20%)**: Test the interaction between two modules (e.g., Orchestrator <-> Trainer) using lightweight fixtures.
*   **System Tests (10%)**: End-to-end runs of the full pipeline (using very small systems/budgets).

### Cycle 01 Testing
*   **Unit**: Verify that `Orchestrator` correctly calls the methods of the injected Mocks.
*   **Integration**: Ensure Configuration is loaded correctly and passed to components.
*   **System**: Run the Mock loop for 10 iterations. Verify no crashes and that the state transitions correctly (Init -> Explore -> Train).

### Cycle 02 Testing
*   **Unit**: Test `AdaptiveExplorationPolicy` logic (e.g., "If high temperature, output high strain").
*   **Integration**: Verify `StructureGenerator` produces valid `ase.Atoms` objects that respect the policy constraints.

### Cycle 03 Testing
*   **Unit**: Test `PeriodicEmbedding` logic (geometry checks). Test Error Handling logic (e.g., "If `scf_error`, does it retry with lower mixing beta?").
*   **Integration**: Run a real (tiny) DFT calculation using a mocked DFT executable or a very cheap calculator (EMT) wrapped in the `DFTManager` interface to verify file I/O.

### Cycle 04 Testing
*   **Unit**: Test `ActiveSetSelector` math (D-optimality) on synthetic matrices.
*   **Integration**: Verify `PacemakerWrapper` correctly constructs the command line arguments and parses the output logs.

### Cycle 05 Testing
*   **Unit**: Test the log parsing logic for LAMMPS (detecting the `halt` signal).
*   **Integration**: Run a short LAMMPS simulation (using a dummy potential) and manually trigger a "high uncertainty" signal to verify the Python handler catches it.

### Cycle 06 Testing
*   **Unit**: Test `Validator` physics checks (e.g., ensure "Imaginary Frequencies" trigger a FAIL).
*   **Integration**: Verify kMC (EON) correctly invokes the potential driver script.
*   **System**: **The Grand Acceptance Test**. Run the Fe/Pt on MgO scenario in "CI Mode" (tiny system). Verify the full chain: Initial Training -> MD Deposition -> Uncertainty Halt -> Re-training -> kMC.
