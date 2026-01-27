# System Architecture: PyAcemaker

## 1. Summary

The **PyAcemaker** project represents a transformative initiative in the field of computational materials science, specifically targeting the democratisation and automation of Machine Learning Interatomic Potential (MLIP) construction. In the contemporary landscape of atomistic modelling, the demand for simulations that combine the high accuracy of *ab initio* quantum mechanical calculations (such as Density Functional Theory - DFT) with the computational efficiency of classical force fields is growing exponentially. MLIPs bridge this gap, but their creation has historically been a craft reserved for a small elite of experts who possess deep, intersecting knowledge of condensed matter physics, statistical mechanics, and data science. The traditional workflow involves a tedious, manual, and error-prone cycle: generating trial atomic structures, performing expensive DFT calculations, fitting a machine learning model, validating it, and then iterating. This process is not only time-consuming—often taking months for a single material system—but is also fraught with pitfalls such as sampling bias, overfitting, and catastrophic extrapolation errors when the potential is used in molecular dynamics (MD) simulations.

PyAcemaker aims to shatter these barriers by providing a "Zero-Config" automated system that leverages the state-of-the-art **Pacemaker** engine, which is based on the Atomic Cluster Expansion (ACE) formalism. The primary goal is to empower a domain researcher—who may not be an ML expert—to input a simple material composition (e.g., "Titanium Oxide") and receive a production-grade, validated potential within days of machine time, with zero human intervention.

The system is architected as an autonomous **Active Learning** loop. Unlike static approaches that rely on pre-computed databases (which are often bloated with redundant data or missing critical high-energy configurations), PyAcemaker dynamically explores the vast chemical and structural phase space. It employs an intelligent agent (the Structure Generator) to propose candidate structures, which are then evaluated for "uncertainty" by the current model. Only those structures that provide significant new information—maximizing the D-Optimality of the training set—are sent to the "Oracle" (DFT engine) for ground-truth calculation. This strategy targets a reduction in computational cost by an order of magnitude compared to random sampling.

Furthermore, PyAcemaker addresses the critical issue of **robustness**. MLIPs are notorious for behaving unphysically (e.g., predicting attractive forces between overlapping atomic cores) in regions where they have no training data. PyAcemaker implements a rigorous "Physics-Informed" approach, utilizing Delta Learning to fit the ML model as a correction to a robust baseline potential (such as Lennard-Jones or ZBL). This ensures that even in the most extreme conditions, the simulation will not crash due to numerical instabilities, allowing the system to "fail gracefully" and trigger self-healing routines.

By integrating Molecular Dynamics (via LAMMPS) and Kinetic Monte Carlo (via EON) into a single cohesive framework, PyAcemaker is not just a training tool but a complete lifecycle management system for interatomic potentials, scalable from a single workstation to massive High-Performance Computing (HPC) environments.

## 2. System Design Objectives

The architectural decisions for PyAcemaker are strictly guided by four primary objectives, which serve as the definitive success metrics for the project.

**1. Zero-Config Workflow (Automation & Usability)**
The foremost objective is to eliminate the complexity barrier. Current tools require users to write complex Python scripts, manage file formats manually, and tune hyperparameters for learning. PyAcemaker's design philosophy is "Convention over Configuration". The system must accept a single, declarative YAML configuration file as input. This file defines *what* is desired (e.g., "A potential for Al-Cu alloys accurate to 1 meV/atom") rather than *how* to do it. The system must autonomously infer reasonable defaults for hyperparameters (cutoff radii, basis set sizes, learning rates) and handle the entire pipeline: initial random structure generation, submission and monitoring of DFT jobs, potential fitting, iterative active learning, and final physics validation. The user experience should be as simple as `mlip-auto start config.yaml`, transforming weeks of human effort into a purely computational task.

**2. Data Efficiency via Active Learning**
A major bottleneck in MLIP construction is the cost of DFT calculations. A naive approach of "sampling everything" leads to datasets containing thousands of structures that are physically redundant (highly correlated), wasting millions of CPU hours. PyAcemaker aims to maximize **Data Efficiency**. It utilizes sophisticated Active Learning strategies, specifically **D-Optimality** (via the MaxVol algorithm), to select the most linearly independent subset of atomic environments for labeling. Combined with real-time **Uncertainty Quantification** (tracking the extrapolation grade, $\gamma$), the system ensures that every DFT calculation contributes maximum information gain to the model. The specific target is to achieve convergence (Energy RMSE < 1 meV/atom, Force RMSE < 0.05 eV/Å) with less than 10% of the DFT calculations required by standard random sampling methods.

**3. Physics-Informed Robustness (Reliability)**
Reliability is paramount. A potential that predicts accurate energies but fails during an MD simulation due to a "segmentation fault" is useless. These failures often occur when atoms approach each other closely (high energy collisions), a regime poorly sampled by standard equilibrium MD. PyAcemaker addresses this by enforcing **Physics-Informed Robustness**. It mandates the use of **Delta Learning**, where the ML model learns the residual energy difference between DFT and a robust physical baseline (ZBL for short-range repulsion, Lennard-Jones for van der Waals). This architecture guarantees that the total potential retains a repulsive core, preventing atoms from collapsing into each other even in unexplored regions of phase space. This allows the Active Learning loop to explore aggressively without fear of unrecoverable simulation crashes.

**4. Scalability, Extensibility, and Modularity**
The system must be built for the future. It must scale from testing on a local laptop to running massive parallel campaigns on HPC clusters (supporting Slurm/PBS schedulers). The architecture must be **Modular**, relying on well-defined interfaces rather than tight coupling. This allows individual components—such as the DFT engine or the Dynamics engine—to be swapped out. For instance, replacing Quantum Espresso with VASP, or LAMMPS with a custom code, should require implementing a single adapter class, not rewriting the core logic. Furthermore, the system is designed to support not just MD (short timescale) but also Kinetic Monte Carlo (long timescale), bridging the gap between nanoseconds and seconds, which is essential for studying diffusion and degradation phenomena.

## 3. System Architecture

The system adopts a **Hub-and-Spoke** architecture, where a central **Orchestrator** acts as the brain, managing state and coordinating the activities of specialized, loosely coupled peripheral modules.

### Component Breakdown

1.  **Orchestrator (The Brain)**
    *   **Function**: The central controller that manages the workflow state machine. It parses the configuration, initializes the project database, and executes the main loop.
    *   **Logic**: It decides when to transition between phases: **Exploration** (generating candidates), **Selection** (choosing the best candidates), **Calculation** (getting ground truth), **Training** (fitting the model), and **Validation** (testing). It persists the `WorkflowState` to disk, ensuring the pipeline can be paused and resumed at any point without data loss.

2.  **Structure Generator (The Explorer)**
    *   **Function**: Responsible for generating candidate atomic structures.
    *   **Logic**: It implements an **Adaptive Exploration Policy**. Initially, it may use "Cold Start" methods like random cell distortion (rattling) and scaling to sample the Equation of State. As the potential improves, it switches to "Hot Exploration" using MD or MC simulations driven by the current potential to find high-uncertainty regions (e.g., high-temperature liquids, defect structures, surfaces).

3.  **Oracle (The Judge)**
    *   **Function**: Provides the "Ground Truth" data by performing Density Functional Theory (DFT) calculations.
    *   **Logic**: This module encapsulates the complexity of running codes like **Quantum Espresso**. Crucially, it includes a **Self-Healing** mechanism. If a DFT calculation fails (e.g., SCF convergence error), the Oracle automatically attempts to fix it by adjusting parameters (mixing beta, smearing width, diagonalization algorithm) before giving up. It also handles **Periodic Embedding**, creating isolated training clusters from bulk simulations.

4.  **Trainer (The Learner)**
    *   **Function**: Fits the Machine Learning Potential to the accumulated data.
    *   **Logic**: It wraps the **Pacemaker** engine. It prepares the training dataset, filtering it through the **Active Set** selection (D-optimality) to remove redundant structures. It then executes the fitting process, ensuring the **Delta Learning** baseline is correctly subtracted from the training data, and manages the output `potential.yace` files, versioning them for tracking.

5.  **Dynamics Engine (The Executor)**
    *   **Function**: Runs simulations to explore the phase space and test the potential.
    *   **Logic**: This module interfaces with **LAMMPS** (for Molecular Dynamics) and **EON** (for Kinetic Monte Carlo). Its most critical feature is **On-the-Fly (OTF) Monitoring**. It uses a "Watchdog" (via `fix halt`) to monitor the max extrapolation grade ($\gamma$) at every step. If uncertainty spikes, it halts the simulation immediately, saving the state for the Orchestrator to inspect.

6.  **Validator (The Gatekeeper)**
    *   **Function**: Assesses the physical validity of the generated potential.
    *   **Logic**: It goes beyond simple test-set errors (RMSE). It calculates fundamental physical properties: **Phonon Dispersion** (checking for dynamic stability/imaginary frequencies), **Elastic Constants** (checking Born stability criteria), and **Equation of State** (checking bulk modulus). It acts as a quality gate, preventing physically unsound potentials from being deployed.

### Architectural Diagram

```mermaid
graph TD
    User[User] -->|Config.yaml| Orch[Orchestrator]

    subgraph "Core Loop"
        Orch -->|1. Request Structures| Gen[Structure Generator]
        Gen -->|2. Candidate Structures| Orch

        Orch -->|3. Submit Jobs| Oracle[Oracle (DFT)]
        Oracle -->|4. Forces & Energies| DB[(Database)]

        DB -->|5. Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. Potential.yace| Dyn[Dynamics Engine]

        Dyn -->|7. Run MD/kMC| Dyn
        Dyn -- Halted (High Uncertainty) --> Orch
    end

    subgraph "Validation"
        Trainer -->|Candidate Potential| Val[Validator]
        Val -->|Pass/Fail| Orch
    end

    Dyn -->|Final Model| Deploy[Production]
```

## 4. Design Architecture

The detailed design of PyAcemaker relies heavily on modern Python practices, specifically utilizing **Pydantic** for rigorous data validation and schema definition. The goal is to catch configuration errors and logic bugs "Fail Fast," before expensive calculations are launched.

### File Structure & Organization

The codebase is organized into semantically meaningful modules under `src/mlip_autopipec/`.

```
mlip_autopipec/
├── orchestrator/           # Workflow management & State Machine
│   ├── loop.py             # Main execution loop
│   ├── state.py            # Serializable State models
│   └── workflow.py         # Task dependency management
├── config/                 # Pydantic Schemas (The Contract)
│   ├── main_config.py      # Root configuration
│   ├── dft_config.py       # DFT parameters
│   └── training_config.py  # Pacemaker parameters
├── generator/              # Structure creation logic
│   ├── defects.py          # Point defect generation
│   ├── explorer.py         # Random/MD-driven exploration strategies
│   └── policy.py           # Adaptive decision logic
├── dft/                    # Oracle implementation
│   ├── quantum_espresso.py # QE input/output handling
│   └── manager.py          # Batch job management & Self-Healing
├── trainer/                # Pacemaker interface
│   ├── pace.py             # Wrapper for pace_train
│   └── active_set.py       # Wrapper for pace_activeset
├── dynamics/               # MD/kMC engines
│   ├── lammps/             # LAMMPS interface & Log parsing
│   └── eon/                # EON interface for kMC
├── validator/              # Physics validation suite
│   ├── phonons.py          # Phonopy integration
│   ├── elastic.py          # Elastic tensor calculation
│   └── eos.py              # Equation of State fitting
└── database/               # Data persistence layer
    └── manager.py          # ASE Atoms storage (pickled/database)
```

### Data Models & Class Definitions

The system is underpinned by a set of strict Data Models that define the "Language" of the application.

*   **`WorkflowConfig`**: The root configuration object. It strictly validates the user's YAML input, ensuring, for example, that the `pseudopotential_dir` exists, that `cutoff` is positive, and that `elements` are valid strings. It nests specialized sub-configs like `DFTConfig` and `TrainingConfig`.
*   **`WorkflowState`**: A JSON-serializable object that tracks the runtime state. It stores the `cycle_index`, the `current_phase` (e.g., `CALCULATION`), paths to the latest potentials, and the history of validation metrics. This allows the Orchestrator to be stateless between runs, enabling resume functionality.
*   **`StructureMetadata`**: An extension of the standard ASE `Atoms.info` dictionary. It captures the **Provenance** of every structure: where did it come from? (e.g., "High-T MD Halt frame 1024"), what was the uncertainty score?, and has it been validated?
*   **`DFTResult`**: A standardized data container for the output of the Oracle. It abstracts away the specific output formats of QE or VASP, providing a unified interface (`energy`, `forces`, `stress`, `converged`) for the rest of the system.

**Key Classes**:
*   **`Orchestrator`**: The entry point. Implements the `run()` method which contains the high-level `while not converged:` loop.
*   **`DFTManager`**: Implements the `submit_batch()` method. It uses the **Strategy Pattern** to apply different `SelfCorrectionStrategy` objects (e.g., "ReduceMixing", "IncreaseSmearing") when errors occur.
*   **`LammpsRunner`**: Manages the `subprocess` calls to LAMMPS. It is responsible for injecting the `fix halt` commands and the `pair_style hybrid/overlay` commands into the input stream.

## 5. Implementation Plan

The development is strictly governed by the **AC-CDD (Architectural-Centric Cycle-Driven Development)** methodology, dividing the project into 6 distinct, testable cycles.

### CYCLE 01: Core Framework & Oracle
*   **Objective**: Establish the project skeleton, configuration system, and the "Oracle" capable of generating ground truth.
*   **Details**: Setup the Python package structure and dependencies (`pyproject.toml`). Implement the Pydantic schemas for configuration. Implement the `QERunner` class to generate `pw.in` files from ASE Atoms and parse `pw.out`. Implement the `DFTManager` with basic batch processing.
*   **Deliverable**: A CLI tool that can read a config and run a static DFT calculation on a provided structure structure file.

### CYCLE 02: Structure Generation & Database Management
*   **Objective**: Enable the creation of training candidates and the management of atomic data.
*   **Details**: Implement the `StructureGenerator` with "Cold Start" capabilities (Random Rattling, Strain). Implement the `DatabaseManager` to persist ASE Atoms objects efficiently (using compressed pickle format or ExtXYZ). Implement the basic `Trainer` wrapper for Pacemaker to perform a simple fit on a static dataset.
*   **Deliverable**: A pipeline that can generate 100 random structures, save them to a database, and run a dummy training process to produce a `potential.yace`.

### CYCLE 03: Dynamics Engine (LAMMPS MD) & Uncertainty
*   **Objective**: Implement the inference engine with robust safety mechanisms.
*   **Details**: Implement the `LammpsRunner`. Crucially, implement the `HybridPotential` logic to automatically generate `pair_style hybrid/overlay` inputs (ACE + ZBL). Implement the `UncertaintyMonitor` using the `fix halt` command in LAMMPS to trigger stops when $\gamma$ exceeds the threshold. Implement the `LogParser` to detect these halts.
*   **Deliverable**: A tool that runs an MD simulation which automatically stops (halts) when the potential becomes unreliable, without crashing the software.

### CYCLE 04: Active Learning Loop (Integration)
*   **Objective**: Close the loop. Connect Dynamics, Oracle, and Trainer into an autonomous cycle.
*   **Details**: Implement the `Orchestrator` main loop logic. Implement the `SelectionStrategy` (Periodic Embedding) to extract local clusters from halted MD frames and prepare them for DFT. Implement `ActiveSetSelection` (D-optimality) to filter structures. Enable the system to Retrain the potential and Resume MD.
*   **Deliverable**: A fully functional autonomous loop that starts from scratch, runs MD, encounters uncertainty, learns from it, and improves the potential over time.

### CYCLE 05: Validation Framework
*   **Objective**: Ensure the generated potential is physically valid and publication-ready.
*   **Details**: Implement the `Validator` module. Integrate `phonopy` to check for dynamic stability (imaginary phonons). Implement scripts for Elastic Constants ($C_{ij}$) and Equation of State (EOS) calculations. Create a reporting system (HTML/Markdown) to visualize these properties.
*   **Deliverable**: The system can reject a fitted potential that is numerically accurate (low RMSE) but physically broken (e.g., unstable crystal structure).

### CYCLE 06: Advanced Dynamics (kMC) & Adaptive Policy
*   **Objective**: Extend capabilities to long timescales and intelligent exploration strategies.
*   **Details**: Implement the `EONWrapper` to integrate Kinetic Monte Carlo for rare event sampling. Implement the `AdaptivePolicy` engine, which analyzes the run history to dynamically adjust exploration parameters (e.g., "We need more high-temperature data" or "We need to sample defects"). Final Polish of the system.
*   **Deliverable**: The complete PyAcemaker system, capable of handling complex exploration tasks autonomously and efficienty.

## 6. Test Strategy

The testing strategy is paramount to ensuring the reliability of an automated scientific workflow. We employ a rigorous "Pyramid Testing" approach.

**1. Unit Testing (The Foundation)**
Every individual module (`Oracle`, `Trainer`, `Generator`) will be accompanied by a comprehensive suite of unit tests using `pytest`.
*   **Mocking**: We will heavily use `unittest.mock`. External executables (LAMMPS, QE, Pacemaker) will be mocked. For example, `QERunner` tests will not call `pw.x` but will verify that the generated `pw.in` string is correct and that the parser correctly extracts data from a pre-provided `pw.out` file.
*   **Schema Validation**: Tests will verify that `WorkflowConfig` correctly rejects invalid YAML files (e.g., negative cutoffs) with helpful error messages.
*   **Coverage**: We aim for high code coverage (>80%) for the core logic in `orchestrator` and `database` to ensure business logic is sound.

**2. Integration Testing (The Glue)**
Integration tests will verify the interfaces *between* modules.
*   **Data Flow**: We will verify that a structure object generated by `StructureGenerator` can be successfully saved by `DatabaseManager`, loaded by `Trainer`, and converted into a Pacemaker dataset without data loss.
*   **IO Tests**: We will verify that `LammpsRunner` generates physically correct input files (checking for the presence of `hybrid/overlay`) and that `LogParser` correctly identifies "Halt" events from log files.
*   **Mocked Loop**: We will run a "dry run" of the Active Learning loop where the DFT and Training steps are mocked (returning dummy energies and dummy potentials). This allows us to verify the *logic* of the state machine (transitions between phases) without the computational cost.

**3. System / End-to-End (E2E) Testing (The Verification)**
For each cycle, a "User Acceptance Test" (UAT) will be defined and executed.
*   **Real Binaries**: These tests require the actual binaries (LAMMPS, QE, Pacemaker) to be installed in the CI or local environment.
*   **Scenarios**: We will define realistic scenarios (e.g., "Generate a potential for Aluminum").
*   **Verification**: The E2E tests will run a shortened version of the workflow (e.g., 1 iteration, small system, loose convergence criteria) to ensure the system actually works from a user perspective and produces a runnable potential.

**4. Validation Logic Regression Testing**
Specific attention will be paid to the `Validator` module. We will maintain a set of "Gold Standard" potentials (known good) and "Broken" potentials (known bad/unstable). The test suite must confirm that the Validator correctly marks the good ones as PASS and the bad ones as FAIL. This ensures that the safety gates of the system are functioning correctly.
