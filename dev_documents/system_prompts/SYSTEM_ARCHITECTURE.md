# System Architecture & Design Document

## 1. Summary

The **MLIP-AutoPipe** (Machine Learning Interatomic Potentials - Automated Pipeline) is a state-of-the-art, "Zero-Human" protocol designed to revolutionise the way materials scientists generate, train, and validate machine learning potentials. In the traditional paradigm of materials simulation, there exists a fundamental trade-off between accuracy and scale. First-principles methods, such as Density Functional Theory (DFT), offer high accuracy but are computationally prohibitively expensive, scaling cubically ($O(N^3)$) with the number of atoms. Conversely, classical Molecular Dynamics (MD) simulations can handle millions of atoms but suffer from limited accuracy due to empirical potential functions. Machine Learning Interatomic Potentials (MLIPs) bridge this gap, promising DFT-level accuracy at MD-level speeds. However, the construction of robust MLIPs faces a "chicken-and-egg" problem: generating high-quality training data requires sampling diverse phase spaces, but exploring these spaces requires an already competent potential to prevent physical absurdity during sampling.

MLIP-AutoPipe solves this by implementing a fully autonomous Active Learning Loop that iteratively improves the potential without human intervention. The system is architected around five core modules: a Physics-Informed Generator (Module A), a Surrogate Explorer (Module B), an Automated DFT Factory (Module C), a Training Engine (Module D), and a Scalable Inference Engine (Module E).

The workflow begins with a "Cold Start" phase where physical intuition is codified into the system. The Generator creates initial structures using Special Quasirandom Structures (SQS) for alloys, Normal Mode Sampling (NMS) for molecules, and defect engineering strategies. Crucially, these structures are filtered by a pre-trained "Surrogate" foundation model (like MACE-MP or M3GNet) to eliminate physically invalid configurations before expensive DFT calculations are attempted. This "Surrogate-First" strategy dramatically reduces computational waste.

Once the initial dataset is generated, the system enters the "Active Learning" phase. The Inference Engine runs massive MD simulations using the current MLIP. It utilises uncertainty quantification (via extrapolation grades or ensemble variance) to identify atomic environments where the potential's predictions are unreliable. These "high-uncertainty" configurations are extracted using a novel "Periodic Embedding" technique, which cuts out a local cluster and wraps it in a buffer zone with periodic boundary conditions, ensuring the quantum mechanical forces are calculated in a bulk-like environment. These candidates are sent to the DFT Factory, which executes Quantum Espresso calculations with robust auto-recovery logic to handle convergence failures. The new data is added to the database, and the Pacemaker engine retrains the potential. This cycle repeats until the potential reaches a convergence criterion, enabling "Zero-Human" production of production-grade force fields.

## 2. System Design Objectives

The primary objective of MLIP-AutoPipe is to democratise access to high-accuracy materials simulations by abstracting away the complexity of potential generation. The specific design goals and success criteria are as follows:

### 2.1. "Zero-Human" Intervention
The system must operate autonomously from the initial "Minimal Config" input (composition, temperature range, goal) to the final validated potential. This implies robust error handling at every stage.
-   **Auto-Recovery**: The DFT engine must automatically detect convergence failures (e.g., "convergence not achieved", "diagonalization error") and apply heuristic fixes, such as mixing beta reduction, algorithm switching (Davidson to Conjugate Gradient), or electronic temperature elevation.
-   **Self-Correction**: If the active learning loop generates "garbage" structures that cause DFT crashes, the system must identify the source, blacklist the region of phase space, or tighten the uncertainty thresholds.

### 2.2. Computational Efficiency (The "Surrogate-First" Approach)
A major constraint is the high cost of DFT. The system aims to maximise the "Information per CPU-hour" ratio.
-   **Filter before Compute**: By using a Surrogate Model (MACE-MP) to scout the energy landscape, we ensure that only diverse and physically relevant structures (selected via Farthest Point Sampling) are sent to DFT.
-   **Static Calculation Only**: We strictly avoid DFT geometry relaxations (`calculation='relax'`) during training data generation. Instead, we compute static forces on diverse snapshots. This prevents the "collapse" of high-energy configurations which are crucial for training robust potentials.

### 2.3. Accuracy and Scalability
-   **Bulk Accuracy**: The "Periodic Embedding with Force Masking" technique is a critical design objective. It ensures that local environments extracted from large-scale MD are treated as bulk systems in DFT, avoiding surface effects. The "Force Masking" feature allows us to train only on the central atoms of the embedded cluster, ignoring the buffer zone affected by the artificial boundary.
-   **Exascale Readiness**: The architecture must support massive parallelism. The Inference Engine (LAMMPS) and DFT Factory (Quantum Espresso) must run on separate, asynchronous queues. The database must handle concurrent writes from hundreds of workers without locking issues.

### 2.4. Extensibility and Modularity
The system is designed with a "Plugin Architecture". While the default backend uses Quantum Espresso and Pacemaker, the interfaces (Runners) are abstract.
-   **Interchangeable Engines**: It should be possible to swap Quantum Espresso for VASP or CASTEP, or Pacemaker for NequIP or Allegro, by implementing a new `Runner` class, without changing the core orchestration logic.
-   **Strict Typing**: We utilise Pydantic for all data models to ensure strict type validation and schema generation, preventing "silent failures" caused by misconfigured parameters.

## 3. System Architecture

The system follows a micro-modular architecture orchestrated by a central workflow manager. Data flows cyclically between generation, simulation, calculation, and training.

### 3.1. Component Diagram

```mermaid
graph TD
    User[User Input: Minimal Config] -->|defines system| Config[Configuration Manager]
    Config -->|initialises| Gen[Module A: Generator]

    subgraph "Phase 1: Cold Start"
        Gen -->|SQS/NMS/Defects| Surrogate[Module B: Surrogate Explorer]
        Surrogate -->|MACE-MP Eval| Filter[Farthest Point Sampling]
        Filter -->|Selected Candidates| DB[(ASE Database)]
    end

    subgraph "Phase 2: The DFT Factory"
        DB -->|Pending Structures| DFT[Module C: Automated DFT Factory]
        DFT -->|Quantum Espresso| Recovery[Auto-Recovery Logic]
        Recovery -->|Retry Parameters| DFT
        DFT -->|Forces/Stresses| Parser[Output Parser]
        Parser -->|Verified Data| DB
    end

    subgraph "Phase 3: Training & Active Learning"
        DB -->|Completed Data| Train[Module D: Training Engine]
        Train -->|Pacemaker| Potential[Current Potential (.yace)]

        Potential -->|Update| Inference[Module E: Inference Engine]
        Inference -->|LAMMPS MD| Uncertainty[Uncertainty Checker]
        Uncertainty -->|High Error > Threshold| Embedding[Periodic Embedding & Masking]
        Embedding -->|New Candidates| DB
        Uncertainty -->|Low Error| Analysis[Physical Analysis]
    end

    Analysis -->|Results| Report[Final Report]
```

### 3.2. Data Flow Description

1.  **Configuration**: The user provides a YAML file defining the elements, composition, and simulation goals (e.g., "Fe-Ni alloy, 300K-1500K").
2.  **Generation (Cold Start)**: The `Generator` creates thousands of candidate structures covering the composition space (SQS), elastic deformations (Strain), and thermal displacements (Rattling).
3.  **Surrogate Filtering**: The `Surrogate Explorer` uses a pre-trained MACE model to predict energies. It discards exploded structures and uses Farthest Point Sampling (FPS) to select the most diverse 10% for DFT.
4.  **DFT Execution**: The `DFT Factory` picks up candidates. It runs Quantum Espresso with strict settings (SCF only, `tprnfor=true`). If a job fails, the `RecoveryHandler` modifies input parameters (e.g., `mixing_beta`) and retries.
5.  **Training**: The `Training Engine` exports valid data from the DB to `.extxyz` format and runs Pacemaker. It optimises the potential using a Delta-Learning approach (ZBL reference).
6.  **Active Learning Loop**: The new potential is loaded into the `Inference Engine`. LAMMPS runs MD simulations. If the `Extrapolation Grade` ($\gamma$) exceeds a threshold (e.g., 5.0), the simulation pauses. The problematic snapshot is extracted, wrapped in a periodic buffer, and sent back to the DFT queue.

## 4. Design Architecture

The codebase is structured to enforce separation of concerns, using a clear directory hierarchy and strict Pydantic data models.

### 4.1. File Structure

```ascii
mlip_autopipec/
├── app.py                      # CLI Entrypoint (Typer)
├── config/                     # Configuration Management
│   ├── models.py               # Aggregated Config Models
│   └── schemas/                # Granular Pydantic Schemas
│       ├── dft.py              # QE Parameters
│       ├── training.py         # Pacemaker Parameters
│       └── workflow.py         # Orchestration Settings
├── data_models/                # Core Data Structures
│   ├── atoms.py                # ASE Atoms Pydantic Wrappers
│   ├── manager.py              # Database Abstraction Layer
│   └── status.py               # Enum Definitions (Pending, Running, etc.)
├── generator/                  # Module A: Structure Generation
│   ├── builder.py              # SQS & Supercell Builder
│   ├── defects.py              # Point Defect Injector
│   └── transformations.py      # Strain & Rattle Logic
├── surrogate/                  # Module B: Surrogate Model
│   ├── mace_wrapper.py         # Interface to MACE-MP
│   └── sampling.py             # Farthest Point Sampling
├── dft/                        # Module C: DFT Factory
│   ├── inputs.py               # Input File Generator
│   ├── runner.py               # QE Execution & Monitoring
│   ├── recovery.py             # Error Handling Strategies
│   └── parsers.py              # Output Parsing & Validation
├── training/                   # Module D: Training Engine
│   ├── dataset.py              # DB to ExtXYZ Converter
│   └── pacemaker.py            # Pacemaker Execution Wrapper
├── inference/                  # Module E: Inference & OTF
│   ├── runner.py               # LAMMPS Runner with Uncertainty
│   └── embedding.py            # Cluster Extraction & Masking
└── orchestration/              # Workflow Management
    ├── workflow.py             # The Main Loop Logic
    └── state.py                # State Persistence (JSON/Pickle)
```

### 4.2. Key Design Patterns
-   **Pydantic for Everything**: All configuration and data exchange between modules happen via Pydantic models. This ensures that a `DFTConfig` object is always valid before it reaches the `QERunner`.
-   **Dependency Injection**: The `WorkflowManager` receives instances of `QERunner`, `PacemakerWrapper`, etc., allowing for easy mocking during tests.
-   **Stateless Runners**: The runners (DFT, Training) are largely stateless; they take an input, execute a process, and return a result. State is managed centrally by `WorkflowState` and the Database.
-   **Database as Queue**: We use the ASE database (`sqlite` or `postgresql`) not just for storage but as a priority queue. A `status` column (`pending`, `running`, `completed`, `failed`) controls the flow.

## 5. Implementation Plan

The project will be implemented in 8 sequential cycles.

### **Cycle 01: Core Framework & Data Models**
-   **Goal**: Establish the project skeleton, configuration system, and database schema.
-   **Features**:
    -   Setup `pyproject.toml` and directory structure.
    -   Implement `MLIPConfig` and sub-configs (DFT, Training) using Pydantic.
    -   Implement `DatabaseManager` wrapping `ase.db` with schema enforcement.
    -   Define `ASEAtoms` custom type for validation.

### **Cycle 02: The Generator (Module A)**
-   **Goal**: Implement the physics-informed structure generation logic.
-   **Features**:
    -   Implement `StructureBuilder` for basic supercells.
    -   Implement `StrainGenerator` for elastic deformations (EOS, Shear).
    -   Implement `RattleGenerator` for thermal noise.
    -   Implement `DefectGenerator` for vacancies and interstitials (using `pymatgen`).

### **Cycle 03: The Surrogate (Module B)**
-   **Goal**: Integrate MACE foundation model for pre-screening.
-   **Features**:
    -   Implement `MaceWrapper` to run inference using `mace-torch`.
    -   Implement `FarthestPointSampling` (FPS) selector to pick diverse structures.
    -   Integrate Surrogate check into the generation pipeline.

### **Cycle 04: DFT Factory Basic (Module C - Part 1)**
-   **Goal**: Basic execution of Quantum Espresso.
-   **Features**:
    -   Implement `InputGenerator` for QE (`pw.x`).
    -   Implement `QERunner` to execute binary commands safely.
    -   Implement `QEOutputParser` to extract Energy, Forces, and Stress.
    -   Note: No error recovery in this cycle.

### **Cycle 05: DFT Factory Advanced (Module C - Part 2)**
-   **Goal**: Robustness and Error Recovery.
-   **Features**:
    -   Implement `RecoveryHandler` to parse QE crash logs.
    -   Implement strategies: `MixingBetaReduction`, `DiagonalizationSwitch`, `TemperatureIncrease`.
    -   Implement "Zombie Job Killer" for timeouts.

### **Cycle 06: Training Engine (Module D)**
-   **Goal**: Automate MLIP training with Pacemaker.
-   **Features**:
    -   Implement `DatasetBuilder` to export DB rows to `.extxyz`.
    -   Implement `PacemakerWrapper` to run `pacemaker` training jobs.
    -   Implement metric parsing (RMSE extraction).

### **Cycle 07: Inference & Active Learning (Module E)**
-   **Goal**: Close the loop with MD and Uncertainty Quantification.
-   **Features**:
    -   Implement `LammpsRunner` for MD simulations.
    -   Implement `UncertaintyChecker` (Extrapolation Grade monitoring).
    -   Implement `EmbeddingExtractor` for Periodic Embedding and Force Masking of high-error clusters.

### **Cycle 08: Orchestration & CLI**
-   **Goal**: Tie everything together into a runnable application.
-   **Features**:
    -   Implement `WorkflowManager` to coordinate the generation -> DFT -> training -> inference loop.
    -   Implement `app.py` CLI with `typer`.
    -   Final Integration Tests.

## 6. Test Strategy

We adopt a strict Test-Driven Development (TDD) approach with `pytest`.

### Cycle 01 Testing
-   **Unit**: Verify `MLIPConfig` raises validation errors on bad input. Verify `DatabaseManager` creates tables and stores atoms correctly.
-   **Integration**: Initialize a DB, write a config, and read it back.

### Cycle 02 Testing
-   **Unit**: Test `StrainGenerator` produces correct tensors. Test `DefectGenerator` creates vacancies without overlapping atoms.
-   **Integration**: Generate a sequence of 100 structures and verify they load in ASE.

### Cycle 03 Testing
-   **Unit**: Mock `MaceWrapper` output to test `FarthestPointSampling` logic (ensure it picks distinct points).
-   **Integration**: Run `MaceWrapper` on a dummy atom (if model available) or strict mock to ensure tensor shapes match.

### Cycle 04 Testing
-   **Unit**: Verify `InputGenerator` creates valid `pw.x` input text. Test parser against sample QE output files.
-   **Integration**: Run a dummy `QERunner` (mocking the binary) to check file I/O operations.

### Cycle 05 Testing
-   **Unit**: Feed simulated error strings (e.g., "convergence not achieved") to `RecoveryHandler` and verify it suggests the correct parameter change.
-   **Integration**: Simulate a failing job loop and verify the runner retries with new parameters.

### Cycle 06 Testing
-   **Unit**: Verify `DatasetBuilder` creates valid `.extxyz` formatting (especially properties).
-   **Integration**: Mock `pacemaker` execution and verify the wrapper captures stdout/stderr.

### Cycle 07 Testing
-   **Unit**: Test `EmbeddingExtractor` correctly wraps coordinates (periodic boundaries). Test Force Masking array generation.
-   **Integration**: Run a short LAMMPS mock loop and trigger the uncertainty threshold.

### Cycle 08 Testing
-   **System**: Run a "Dry Run" of the full loop using mocks for all heavy engines (QE, Pacemaker, MACE). Verify data flows from Generator to DB to Training to Inference and back.
