# System Architecture: MLIP-AutoPipe

## 1. Summary

The **MLIP-AutoPipe** (Machine Learning Interatomic Potential Automated Pipeline) represents a paradigm shift in the computational materials science domain. Traditionally, the construction of robust machine learning potentials (MLIPs) has been a bespoke, artisanal process requiring significant manual intervention by domain experts. This "Zero-Human" autonomous pipeline aims to democratize access to high-fidelity molecular dynamics simulations by automating the end-to-end workflow—from the initial generation of diverse atomic structures to the active learning loops that refine the potential's accuracy.

The project addresses the fundamental **"Chicken and Egg" problem** in MLIP development: to generate high-quality training data (DFT calculations on diverse configurations), one needs a simulation engine capable of exploring the phase space without unphysical crashing; however, to build such an engine, one needs the very training data that is yet to be generated. If one simply runs DFT on random structures, the data is too high-energy and irrelevant. If one runs MD with a bad potential, the simulation explodes. **MLIP-AutoPipe** solves this by employing a multi-staged, bootstrap approach. It begins with physics-informed random structure generation (SQS, Normal Mode Sampling), leverages pre-trained "foundation models" (like MACE-MP) as surrogates to scout the potential energy surface, and then iteratively refines a specialized local potential (ACE) using active learning.

The target audience for this system includes **Materials Scientists** and **Computational Physicists** who require predictive accuracy for complex systems (such as high-entropy alloys, molecular crystals, or defect-laden interfaces) but lack the resources to manually curate datasets. By providing a declarative configuration interface (e.g., "Target: Fe-Ni Alloy, Goal: Melt-Quench"), users can trigger a massive, asynchronous computational campaign. The system autonomously manages thousands of First-Principles (DFT) calculations, handles convergence failures, trains the potential, runs molecular dynamics simulations, detects uncertainty, and closes the loop by requesting new ground-truth data—all without human supervision.

Technically, the system is architected as a modular, event-driven distributed system. It adheres to strict software engineering principles including **AC-CDD (Architectural Code-Construction Driven Design)**, rigorous type safety (via Pydantic and Mypy), and comprehensive testing. It supports scalability from a single workstation to large-scale High-Performance Computing (HPC) clusters using Dask for task orchestration. The ultimate vision is to reduce the time-to-solution for new potentials from months of manual effort to days of autonomous compute.

Furthermore, the system is designed to be **future-proof**. By decoupling the "Generator", "Surrogate", "Calculator", and "Trainer" modules, we ensure that new developments in the field (e.g., a better Foundation Model replacing MACE, or a faster DFT code replacing Quantum Espresso) can be integrated with minimal refactoring. The use of a strict Pydantic schema as the "Lingua Franca" of the system ensures that these modules can communicate regardless of their internal implementation details.

In summary, MLIP-AutoPipe is not just a script runner; it is an intelligent agent that understands the physics of simulation failures and knows how to recover from them, turning the "Art" of potential fitting into a reliable "Industrial Process".

## 2. System Design Objectives

### 2.1 Functional Objectives
The primary functional objective is **Zero-Human Intervention**. Once the user defines the material system (e.g., "Al-Cu Alloy") and the simulation goal (e.g., "Calculate Melting Point"), the system must perform every subsequent step autonomously. This includes:
*   **Self-Healing**: The system must detect and recover from at least 90% of standard computational failures. For example, if a DFT calculation fails to converge, the system should automatically apply a recovery strategy (mixing beta reduction, temperature increase) without user input.
*   **Active Learning**: The system must be able to identify "knowledge gaps". It should not just run MD; it should monitor the uncertainty of its predictions. When uncertainty is high, it must pause, excise the problematic configuration, and request ground-truth validation.
*   **Provenance Tracking**: Every data point in the final training set must be traceable. We must know exactly which generation it came from, which surrogate selected it, and which DFT parameters were used to calculate it.

### 2.2 Non-Functional Objectives
*   **Scalability**: The system must scale linearly with available compute resources. Whether running on a 16-core workstation or a 10,000-core SLURM cluster, the orchestration logic (Dask) should efficiently distribute tasks. The "DFT Factory" module must be able to handle queues of 10,000+ pending calculations without memory leaks or database locking issues.
*   **Reliability**: Long-running workflows (spanning days or weeks) are prone to interruptions (power outages, wall-time limits). The system must be **stateless** in its execution logic and **stateful** in its persistence. If the process is killed at any point, restarting it should seamlessly resume from the last checkpoint, checking the database to see what has already been done.
*   **Maintainability**: The codebase should be approachable for both software engineers and domain scientists. We enforce strict typing (mypy strict), comprehensive documentation, and a modular "Plugin" architecture. Pydantic models serve as the contract between modules, preventing "spaghetti code" where dictionaries are passed around blindly.

### 2.3 Business Objectives
*   **Democratization**: Lower the barrier to entry for high-accuracy simulations. A PhD student should be able to train a production-grade potential in their first week, rather than spending 6 months learning the intricacies of fitting codes.
*   **Resource Efficiency**: Maximize the "Information Gain per Core-Hour". DFT is expensive. By using the Surrogate-First strategy (Module B), we ensure that we only spend DFT time on structures that are mathematically guaranteed to improve the potential (via Farthest Point Sampling), rather than recalculating known equilibrium structures repeatedly.
*   **Time-to-Solution**: Reduce the cycle time for new materials discovery. By automating the loop, we eliminate the "human latency" (waiting for a researcher to check a log file, fix an input, and resubmit). The computer works 24/7/365.

## 3. System Architecture

The system follows a modular, event-driven architecture, orchestrated by a central workflow manager. It is conceptualized as a feedback loop containing five distinct processing modules (A through E).

```mermaid
graph TD
    User[User: Minimal Config] --> ConfigFactory[Config Factory]
    ConfigFactory --> SystemConfig[System: Full Execution Config]

    subgraph "Cycle 01: Core"
        SystemConfig --> DB[(ASE Database)]
        DB --> Logger[Logger & Reporter]
    end

    subgraph "Cycle 03: Generator (Module A)"
        SystemConfig --> Generator[Physics-Informed Generator]
        Generator -- SQS/NMS/Defects --> Candidates[Candidate Structures]
    end

    subgraph "Cycle 04: Surrogate (Module B)"
        Candidates --> MACE[MACE-MP Surrogate]
        MACE --> Filter[Pre-Screening]
        Filter --> FPS[Farthest Point Sampling]
        FPS --> Selected[Selected Structures]
    end

    subgraph "Cycle 02: DFT Factory (Module C)"
        Selected --> DFT_Queue
        DFT_Queue --> QERunner[Quantum Espresso Runner]
        QERunner -- Error --> Recovery[Auto-Recovery Logic]
        Recovery --> QERunner
        QERunner -- Success --> DFT_Results[DFT Data (.extxyz)]
        DFT_Results --> DB
    end

    subgraph "Cycle 05: Training (Module D)"
        DB --> Dataset[Training Dataset]
        Dataset --> Pacemaker[Pacemaker Trainer]
        Pacemaker --> Potential[MLIP Potential (.yace)]
    end

    subgraph "Cycle 06-07: Inference (Module E)"
        Potential --> Inference[MD/kMC Engine]
        Inference -- Uncertainty > Threshold --> Extractor[Periodic Embedding]
        Extractor --> Embed_Queue[Embedding Queue]
        Embed_Queue --> DFT_Queue
    end

    subgraph "Cycle 08: Orchestration"
        WorkflowManager[Workflow Manager] --> Generator
        WorkflowManager --> MACE
        WorkflowManager --> QERunner
        WorkflowManager --> Pacemaker
        WorkflowManager --> Inference
    end
```

### Component Interaction Detail
1.  **Configuration Factory**: This is the gateway. It accepts the user's intent (e.g., "I want to study Fe-Ni") and expands it into a rigorous `SystemConfig`. It resolves paths, checks resource availability, and sets up the workspace. It enforces the "Schema-First" design by validating inputs immediately.
2.  **Module A: Generator**: The engine of creativity. It does not run physics; it runs heuristics. It uses Special Quasirandom Structures (SQS) to model alloys and Normal Mode Sampling (NMS) to explore molecular vibrations. It populates the pipeline with "Candidate" structures.
3.  **Module B: Surrogate Explorer**: The filter. It uses a cheap, pre-trained model (MACE-MP) to look at the candidates. It discards the garbage (exploding atoms) and then uses Farthest Point Sampling (FPS) to pick the most diverse subset. This ensures our training data is information-dense.
4.  **Module C: DFT Factory**: The lab. It accepts structures and returns Truth. It wraps Quantum Espresso in a layer of resilience logic. If QE crashes, the factory fixes the input and retries. It guarantees that the database is populated with converged, accurate results.
5.  **Module D: Trainer**: The learner. It takes the database content, calculates the "Delta" (difference between DFT and a ZBL baseline), and fits an ACE potential using Pacemaker. It manages the training hyperparameters automatically.
6.  **Module E: Inference**: The explorer. It takes the trained potential and runs "Risky" simulations (high temperature, defects). It watches the "Extrapolation Grade". If the potential gets confused, the Inference engine pauses, cuts out the confusing part of the simulation, and sends it back to the DFT Factory for clarification.

## 4. Design Architecture

The project adopts a strict "Schema-First" design philosophy. Pydantic models serve as the single source of truth for all data structures, ensuring consistency across module boundaries.

### File Structure (ASCII Tree)
```ascii
mlip_autopipec/
├── src/
│   ├── config/             # Pydantic models for configuration
│   │   ├── __init__.py
│   │   ├── models.py       # All shared config models (Minimal, System)
│   │   └── factory.py      # Logic to expand minimal config -> SystemConfig
│   ├── core/               # Core utilities
│   │   ├── database.py     # ASE-db wrapper with schema enforcement
│   │   └── logging.py      # Structured logging setup
│   ├── generator/          # Module A: Structure Generation
│   │   ├── alloy.py        # SQS, Strain, Rattling logic
│   │   └── molecule.py     # Normal Mode Sampling (NMS)
│   ├── surrogate/          # Module B: Pre-screening
│   │   ├── mace_client.py  # Wrapper for MACE-MP inference
│   │   └── sampling.py     # Farthest Point Sampling (FPS) implementation
│   ├── dft/                # Module C: Ground Truth Calculation
│   │   ├── runner.py       # QERunner class
│   │   └── recovery.py     # Error handling state machine
│   ├── training/           # Module D: MLIP Training
│   │   └── pacemaker.py    # Interface to Pacemaker executable
│   ├── inference/          # Module E: MD & Active Learning
│   │   ├── md_engine.py    # LAMMPS wrapper
│   │   └── embedding.py    # Periodic embedding & Force Masking
│   └── orchestration/      # Workflow management
│       └── manager.py      # Dask/Celery orchestration logic
├── tests/
│   ├── unit/
│   └── integration/
├── dev_documents/
│   └── system_prompts/
│       ├── CYCLE01/
│       │   ├── SPEC.md
│       │   └── UAT.md
│       └── ... (02-08)
├── pyproject.toml
└── README.md
```

### Data Models & Contracts
The system relies on a few core Pydantic models:
-   **`MinimalConfig`**: The user-facing contract. Simple, high-level, permissive.
-   **`SystemConfig`**: The machine-facing contract. Precise, fully resolved paths, strict types.
-   **`DFTResult`**: A standardized object encapsulating the outcome of a DFT run (Energy, Forces, Stress, Logs).
-   **`StructureMetadata`**: Attached to every atomic structure in the DB, tracking its provenance (e.g., "Generated by SQS", "Selected by FPS", "Extracted from MD frame 500").

## 5. Implementation Plan

The project execution is divided into 8 distinct, testable cycles. Each cycle delivers a concrete piece of functionality that builds upon the previous ones.

### CYCLE 01: Core Framework & User Interface
**Objective**: Establish the project skeleton and data infrastructure.
**Details**: We will implement the `src/config` module using Pydantic V2 to define the `MinimalConfig` and `SystemConfig` schemas. This ensures type safety from day one. We will also implement the `DatabaseManager` in `src/core`, which wraps `ase.db` to enforce our specific metadata schema (provenance tracking). Finally, we will set up the centralized `Logger` and the basic CLI structure.
**Deliverable**: A functioning CLI command `mlip-auto init` that reads a YAML file, validates it, creates the project directory structure, and initializes an empty SQLite database with the correct schema.

### CYCLE 02: Automated DFT Factory (Module C)
**Objective**: Enable reliable, autonomous DFT calculations.
**Details**: This is the most critical module. We will implement the `QERunner` class to wrap the `pw.x` executable. We will implement the `InputGenerator` to automatically select SSSP pseudopotentials and K-points. Crucially, we will implement the `RecoveryHandler` state machine, which parses QE output for known errors (convergence failure, diagonalization errors) and returns modified input parameters to retry the calculation.
**Deliverable**: A module that can take a list of `ase.Atoms`, run them through Quantum Espresso (handling retries automatically), and return a list of `DFTResult` objects populated with Energy, Forces, and Stress.

### CYCLE 03: Physics-Informed Generator (Module A)
**Objective**: Generate physically relevant initial structures.
**Details**: We will implement the "Cold Start" logic. For alloys, we will use `icet` (or internal logic) to generate Special Quasirandom Structures (SQS). For molecules, we will use Normal Mode Sampling (NMS). We will also implement `DefectGenerator` (vacancies, interstitials). All structures will be subjected to random "Rattling" and "Strain" to broaden the phase space coverage.
**Deliverable**: A `Generator` class that accepts a `TargetSystem` config and returns a diverse list of `Atoms` objects covering the requested composition and distortion space.

### CYCLE 04: Surrogate Explorer (Module B)
**Objective**: Filter and select the best structures from the generator.
**Details**: We will integrate the MACE-MP foundation model. The `SurrogateClient` will use MACE to predict forces for thousands of generated candidates. We will implement a filter to discard "exploding" structures. Then, we will implement the Farthest Point Sampling (FPS) algorithm using SOAP (or ACE) descriptors to select the mathematically most diverse subset of the surviving candidates.
**Deliverable**: A pipeline stage that takes 10,000 raw candidates and returns the "Top 100" most information-rich structures for DFT calculation.

### CYCLE 05: Active Learning & Training (Module D)
**Objective**: Automate the training of ML potentials.
**Details**: We will implement the `PacemakerWrapper`. This class manages the interface with the `pacemaker` training code. It will include a `DatasetBuilder` that queries the ASE database, performs Delta Learning (subtracting ZBL baselines), and formats the data for training. It will also handle the "Active Learning Generations" logic, ensuring we don't overwrite old potentials but create improved versions.
**Deliverable**: A module that takes the populated DFT database and produces a validated `.yace` potential file and a training report (RMSE plots).

### CYCLE 06: Scalable Inference Engine - Part 1 (Module E)
**Objective**: Run basic simulations with the trained potential.
**Details**: We will implement the `LammpsRunner`. This wraps the LAMMPS executable to run MD simulations using the trained `.yace` potential. We will implement the logic to monitor the "Extrapolation Grade" ($\gamma$) via LAMMPS `compute` commands. The runner will be configured to "Mine" for uncertainty—dumping frames only when the potential becomes unreliable.
**Deliverable**: A module capable of running MD simulations (NVT/NPT) and automatically extracting frames that exceed a specified uncertainty threshold.

### CYCLE 07: Scalable Inference Engine - Part 2 (Module E)
**Objective**: Handle complex boundary conditions for re-training data.
**Details**: We will implement the `EmbeddingExtractor` and `ForceMasker`. When MD fails locally, we cannot just cut out a cluster (vacuum errors). We must create a small periodic box around the failure. We will implement the logic to excise this box and, crucially, to calculate the "Force Mask" that tells the trainer to ignore the artificial forces at the box boundaries.
**Deliverable**: A robust algorithm that converts a "Bad MD Frame" into a "Good DFT Input" with appropriate metadata for masked training.

### CYCLE 08: Orchestration & Production Readiness
**Objective**: Integrate all modules into a cohesive, scalable system.
**Details**: We will implement the `WorkflowManager` and `TaskQueue` using Dask. This module will implement the high-level state machine (Generate -> DFT -> Train -> Inference -> Loop). It will handle job dependency management (don't train until DFT is done) and resilience (if 5% of jobs fail, keep going). We will also implement a simple HTML dashboard for monitoring the long-running process.
**Deliverable**: The final `mlip-auto` CLI tool that runs the entire loop autonomously, producing a production-grade potential.

## 6. Test Strategy

### 6.1 General Philosophy
We follow the **Testing Pyramid**:
1.  **Unit Tests (70%)**: Fast, isolated tests for every class and function. We mock all external interactions (Filesystem, Database, External Codes).
2.  **Integration Tests (20%)**: Verifying that modules speak to each other correctly (e.g., Generator output can be read by DFT Runner).
3.  **End-to-End Tests (10%)**: "Grand Mock" tests where we run the full pipeline with physics engines replaced by dummy functions to verify the workflow logic.

### 6.2 Cycle-Specific Strategy

*   **Cycle 01**: Verify that config files are parsed correctly and the DB file is created. Mock filesystem interactions to ensure path resolution works on Windows/Linux.
*   **Cycle 02**: Mock the Quantum Espresso executable. Feed standard output files (success and failure cases) to the parser to verify recovery logic. We will use a library of "known bad" outputs to test every branch of the recovery state machine.
*   **Cycle 03**: Verify that generated structures satisfy physical constraints (e.g., symmetry, bond lengths). Assert that SQS generation matches target stoichiometry exactly.
*   **Cycle 04**: Test FPS by ensuring the selected subset maximizes diversity distance on synthetic data. Compare MACE predictions against known values for simple test cases to ensure the interface works.
*   **Cycle 05**: Mock Pacemaker execution. Check that the training dataset is correctly formatted (ZBL subtraction) and split (Train/Test disjointness). Verify that the `input.yaml` generated for Pacemaker matches the Pydantic config.
*   **Cycle 06**: Test the LAMMPS wrapper. Verify that uncertainty thresholds correctly trigger "stop" events. Parse dummy dump files to ensure the `UncertaintyChecker` works.
*   **Cycle 07**: Visual verification of embedding boxes (generating XYZs for manual inspection). Check that force masks correctly zero out buffer atoms in the array.
*   **Cycle 08**: Full system stress test using a dummy "fast" DFT mock. We will simulate a workflow where "Gen 0" fails, "Gen 1" succeeds, and verify the Orchestrator handles the state transitions and checkpoints correctly.
