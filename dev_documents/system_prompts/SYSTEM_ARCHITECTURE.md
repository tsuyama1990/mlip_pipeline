# System Architecture: PYACEMAKER

## 1. Summary

The **PYACEMAKER** system represents a paradigm shift in the construction and deployment of Machine Learning Interatomic Potentials (MLIPs). Traditionally, the development of high-fidelity potentials, such as those based on the Atomic Cluster Expansion (ACE) formalism, has been the domain of expert computational physicists. This process often involves a laborious, manual iterative cycle of structure generation, Density Functional Theory (DFT) calculation, potential fitting, and validation. This manual approach suffers from several critical bottlenecks: it is time-consuming, prone to human error, and often results in potentials that are brittle when extrapolated to unseen configurations. Furthermore, the lack of standardized workflows leads to "zombie" data—configurations that are calculated but never effectively used—wasting valuable computational resources.

PYACEMAKER addresses these challenges by providing a fully automated, "Zero-Config" workflow that democratizes access to state-of-the-art MLIPs. At its core, the system is designed to be an autonomous agent that navigates the vast chemical and structural space of materials. By integrating an **Adaptive Exploration Policy**, the system intelligently samples the most informative configurations, whether they are equilibrium bulk structures, surfaces, defects, or high-energy transition states. This active learning approach ensures that the potential is trained on a "compact but complete" dataset, minimizing the number of expensive DFT calculations required to achieve a target accuracy.

The system is built upon a modular architecture orchestrated by a central Python controller. This **Orchestrator** manages the entire lifecycle of the potential, from the initial "Cold Start" using universal potentials to the final "Production Deployment" of a robust ACE model. It coordinates the activities of specialized components: the **Structure Generator** (the explorer), the **Oracle** (the ground-truth generator via DFT), the **Trainer** (the learner using Pacemaker), and the **Dynamics Engine** (the verifier and explorer via MD/kMC).

Crucially, PYACEMAKER emphasizes **Physics-Informed Robustness**. It does not treat the potential as a mere black-box regression model. Instead, it enforces physical constraints through a Delta Learning framework, where the MLIP learns the correction to a robust physical baseline (such as a Lennard-Jones or ZBL potential). This ensures that even in regions where data is sparse, the potential behaves physically (e.g., preventing nuclear fusion), thereby guaranteeing the stability of downstream molecular dynamics (MD) simulations.

Furthermore, the system is designed for **Scalability and Extensibility**. It supports a seamless transition from local workstation execution to High-Performance Computing (HPC) environments. The architecture is container-ready, with each component capable of running in isolated environments. The inclusion of an **Adaptive Kinetic Monte Carlo (aKMC)** engine allows the system to bridge the time-scale gap, enabling the exploration of rare events and long-term evolution phenomena that are inaccessible to standard MD.

In essence, PYACEMAKER transforms the art of potential fitting into a rigorous, automated engineering discipline. It empowers materials scientists to focus on the *application* of potentials—discovering new alloys, designing catalysts, or optimizing battery materials—rather than the *construction* of the tools themselves. By lowering the barrier to entry and ensuring high reliability, PYACEMAKER accelerates the discovery cycle in computational materials science.

## 2. System Design Objectives

The design of the PYACEMAKER system is guided by four primary objectives, which serve as the pillars for all architectural decisions and implementation strategies.

### 2.1. Zero-Config Automation (The "One-Click" Ideal)
The most significant barrier to MLIP adoption is the complexity of the workflow. Users must typically manage dozens of scripts, input files, and manual data transfers between different software packages (e.g., LAMMPS, Quantum Espresso, Pacemaker).
**Objective:** To reduce the user interface to a single configuration file (`config.yaml`).
**Success Criteria:**
- A user can define a material system (e.g., "Fe-Pt alloy") and a target accuracy in one file.
- The system automatically handles all intermediate steps: generating initial structures, setting up DFT calculations (pseudopotentials, k-points), managing the active learning loop, and monitoring convergence.
- No manual intervention is required for error handling; the system must self-heal (e.g., restart divergent DFT calculations).

### 2.2. Data Efficiency via Active Learning
DFT calculations are computationally expensive. A naive approach of "random sampling" or "grid search" in configuration space is prohibitively costly and inefficient.
**Objective:** To achieve state-of-the-art accuracy with the minimum number of DFT evaluations.
**Success Criteria:**
- Implement an **Active Learning** loop that selects only the most "informative" structures for labeling.
- Use uncertainty quantification (Extrapolation Grade $\gamma$) to identify gaps in the potential's knowledge.
- Target a 10x reduction in training set size compared to random sampling, while maintaining Energy RMSE < 1 meV/atom and Force RMSE < 0.05 eV/Å.
- Utilize **D-Optimality** (MaxVol algorithm) to prune redundant data from the training set, ensuring the information content is maximized per unit of storage and compute.

### 2.3. Physics-Informed Robustness & Safety
Machine learning models are notorious for unpredictable behavior outside their training domain (extrapolation). In MD simulations, a single unphysical force prediction (e.g., attractive core) can cause the simulation to explode.
**Objective:** To guarantee physical stability and safety in all regimes, including far-from-equilibrium conditions.
**Success Criteria:**
- Implement **Delta Learning**: The ACE potential fits the residual energy ($E_{DFT} - E_{Baseline}$) where the baseline is a robust physics-based potential (LJ/ZBL).
- Enforce **Hard Constraints**: The system must prevent atoms from overlapping (Core Repulsion) regardless of the ML prediction.
- **On-the-Fly (OTF) Reliability**: The Dynamics Engine must monitor the uncertainty $\gamma$ in real-time. If the simulation enters an unknown region, it must safely halt (Graceful Degradation) and trigger a learning cycle, rather than producing garbage data or crashing.

### 2.4. Scalability & Hybrid Simulation (MD + kMC)
Materials phenomena occur across vast spatiotemporal scales. A potential trained only on short MD trajectories may fail to capture long-term diffusion or phase transformations.
**Objective:** To support multi-scale exploration and deployment.
**Success Criteria:**
- **Hybrid Architecture**: Seamlessly integrate Molecular Dynamics (for thermal fluctuations) and Adaptive Kinetic Monte Carlo (for rare events/long timescales).
- **HPC Readiness**: The Oracle and Trainer components must be capable of dispatching jobs to Slurm/PBS schedulers (via internal abstraction or future extension).
- **Containerization**: Components should be loosely coupled to allow them to run in separate containers (Docker/Singularity), facilitating deployment on diverse hardware (local GPU workstations vs. supercomputers).

## 3. System Architecture

The PYACEMAKER system follows a **Hub-and-Spoke** architecture, with the `Orchestrator` acting as the central hub that coordinates the data flow between specialized, loosely coupled components.

### High-Level Component Diagram

```mermaid
graph TD
    subgraph "User Space"
        Config[config.yaml] --> Orch
        Orch --> FinalPot[Final Potential (.yace)]
        Orch --> Report[Validation Report]
    end

    subgraph "Core System (Orchestrator)"
        Orch[Orchestrator]
        State[State Manager]
        Data[Dataset Manager]
    end

    subgraph "Components"
        Gen[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (MD/kMC)]
        Val[Validator]
    end

    %% Data Flow - The Active Learning Cycle
    Orch -- "1. Request Structures" --> Gen
    Gen -- "Candidate Structures" --> Orch

    Orch -- "2. Explore & Detect (OTF)" --> Dyn
    Dyn -- "High Uncertainty Structures" --> Orch

    Orch -- "3. Label Data" --> Oracle
    Oracle -- "Labeled Data (E, F, V)" --> Data

    Data -- "Training Set" --> Trainer
    Trainer -- "Potential (.yace)" --> Orch

    Orch -- "4. Deploy" --> Dyn
    Orch -- "5. Verify" --> Val
    Val -- "Quality Metrics" --> Orch

    %% Feedback Loops
    Val -.->|Fail: Refine| Gen
    Dyn -.->|Halt: Re-train| Oracle
```

### Component Descriptions

1.  **Orchestrator (`src/mlip_autopipec/core/orchestrator.py`)**
    -   **Role**: The brain of the operation. It parses the configuration, manages the workflow state (Cycle 1..N), and invokes other components.
    -   **Responsibility**: Error handling, file system management, and convergence checking.

2.  **Structure Generator (`src/mlip_autopipec/components/generator`)**
    -   **Role**: The explorer. Generates initial random structures, deformed lattices, and surfaces to bootstrap the learning process.
    -   **Features**: Implements `AdaptiveExplorationPolicy` to shift strategy (e.g., from bulk to defects) based on current model maturity.

3.  **Oracle (`src/mlip_autopipec/components/oracle`)**
    -   **Role**: The ground truth provider. Runs DFT calculations (Quantum Espresso/VASP).
    -   **Features**: Self-healing (auto-correction of convergence errors), Periodic Embedding (cutting clusters from MD and embedding them in vacuum/bulk for DFT).

4.  **Trainer (`src/mlip_autopipec/components/trainer`)**
    -   **Role**: The learner. Wraps the `pacemaker` library.
    -   **Features**: Active Set selection (D-Optimality), Delta Learning configuration, and fitting of the ACE B-basis functions.

5.  **Dynamics Engine (`src/mlip_autopipec/components/dynamics`)**
    -   **Role**: The verifier and explorer. Runs MD (LAMMPS) and kMC (EON).
    -   **Features**: On-the-Fly (OTF) uncertainty monitoring (`fix halt`), Hybrid Potential execution (`pair_style hybrid/overlay`).

6.  **Validator (`src/mlip_autopipec/components/validator`)**
    -   **Role**: The gatekeeper. Performs physical validation.
    -   **Features**: Phonon stability checks, Elastic constant calculation, EOS fitting.

## 4. Design Architecture

The system is designed using **Domain-Driven Design (DDD)** principles, enforced by strict Pydantic models. This ensures type safety, configuration validation, and clear interface definitions.

### File Structure Strategy

```ascii
src/mlip_autopipec/
├── components/
│   ├── dynamics/           # Cycle 05
│   │   ├── lammps.py
│   │   └── eon.py
│   ├── generator/          # Cycle 02
│   │   ├── policy.py
│   │   └── builder.py
│   ├── oracle/             # Cycle 03
│   │   ├── qe.py
│   │   └── embedding.py
│   ├── trainer/            # Cycle 04
│   │   └── pacemaker.py
│   └── validator/          # Cycle 06
│       └── metrics.py
├── core/
│   ├── orchestrator.py     # Cycle 06 (Cycle 01 Skeleton)
│   ├── dataset.py          # Cycle 01
│   └── state.py            # Cycle 01
├── domain_models/          # Cycle 01
│   ├── config.py
│   ├── structure.py
│   └── potential.py
├── interfaces/             # Cycle 01
│   └── base_component.py
├── utils/
│   └── logging.py
└── main.py                 # Entry Point
```

### Key Data Models

1.  **`GlobalConfig` (Singleton)**
    -   Defines the entire simulation parameter space.
    -   Immutable during runtime to ensure reproducibility.
    -   Schema: `workdir`, `target_metrics`, `component_configs`.

2.  **`Structure` (Domain Entity)**
    -   A unified representation of atomic configurations (wrapping `ase.Atoms`).
    -   Attributes: `positions`, `numbers`, `cell`, `pbc`, `forces`, `energy`, `stress`, `tags` (for active learning metadata).
    -   Serialization: Custom JSON/YAML handling for Numpy arrays.

3.  **`Potential` (Value Object)**
    -   Represents a trained model artifact.
    -   Attributes: `path_to_yace`, `creation_date`, `training_metrics`, `validation_metrics`.

## 5. Implementation Plan

The project is divided into 6 sequential cycles, adhering to the AC-CDD methodology. Each cycle delivers a testable, working increment of the system.

### CYCLE 01: Core Framework & Mocks
-   **Goal**: Establish the project skeleton, configuration management, and define component interfaces.
-   **Features**:
    -   Setup `pyproject.toml`, directory structure, and logging.
    -   Implement `domain_models` (Config, Structure).
    -   Define abstract base classes (`BaseGenerator`, `BaseOracle`, etc.) in `interfaces/`.
    -   Implement **Mock Components**: `MockGenerator`, `MockOracle` (returns random forces), `MockTrainer` (creates dummy file).
    -   Create a basic `Orchestrator` that runs a "Hello World" loop using mocks.
-   **Deliverable**: A runnable CLI that completes a fake active learning cycle.

### CYCLE 02: Structure Generator
-   **Goal**: Implement intelligent structure creation.
-   **Features**:
    -   `StructureGenerator` implementation using `ase.build` and `pymatgen`.
    -   **Adaptive Exploration Policy**: Logic to decide *what* to generate (Bulk vs Surface vs Cluster).
    -   Rattling (random displacement) and Strain application logic.
-   **Deliverable**: A module that outputs diverse, valid atomic structures based on config.

### CYCLE 03: Oracle (DFT Integration)
-   **Goal**: Enable real physics calculations.
-   **Features**:
    -   `DFTManager` wrapping `ase.calculators.espresso` (Quantum Espresso).
    -   **Self-Healing**: Catch SCF convergence errors and retry with safer params (mixing beta, smearing).
    -   **Periodic Embedding**: Logic to cut a cluster from a large system and embed it in a small box for DFT.
-   **Deliverable**: A robust interface to run DFT calculations and return labeled `Structure` objects.

### CYCLE 04: Trainer (Pacemaker Integration)
-   **Goal**: Enable potential fitting.
-   **Features**:
    -   `PacemakerWrapper` to call `pace_train` and `pace_activeset`.
    -   **Delta Learning**: Setup for `pair_style hybrid` (LJ + ACE).
    -   **Active Set Selection**: Integration of MaxVol algorithm to filter datasets.
-   **Deliverable**: A module that takes labeled structures and outputs a valid `.yace` file.

### CYCLE 05: Dynamics Engine (OTF Loop)
-   **Goal**: Enable feedback and exploration.
-   **Features**:
    -   `LAMMPS` integration via python interface or subprocess.
    -   **OTF Monitoring**: Implementation of `fix halt` based on uncertainty $\gamma$.
    -   **Hybrid Potential**: Logic to write `in.lammps` with `pair_style hybrid/overlay`.
    -   `EON` integration for kMC (optional/stubbed if binaries missing).
-   **Deliverable**: A system that runs MD, detects uncertainty, and requests new data.

### CYCLE 06: Validation & Full Orchestration
-   **Goal**: Complete the loop and ensure quality.
-   **Features**:
    -   `Validator` component: Phonon dispersion, Elastic constants.
    -   Finalize `Orchestrator` logic: Full state management, error recovery, file organization.
    -   Integration of all previous cycles into the seamless "Zero-Config" workflow.
-   **Deliverable**: The production-ready PYACEMAKER system.

## 6. Test Strategy

Testing is continuous and multi-layered. We employ a "Mock vs. Real" strategy to ensure CI/CD compatibility.

### CYCLE 01: Core & Mocks
-   **Unit Tests**: Verify Pydantic models validation (e.g., config checks), Interface compliance.
-   **Integration Tests**: Run the Orchestrator with `Mock` components. Verify the loop completes without crashing.
-   **Strategy**: This is the "Skeleton Test". It proves the wiring is correct.

### CYCLE 02: Structure Generator
-   **Unit Tests**: Verify generated structures have correct stoichiometry, cell size, and reasonable atomic distances (no overlap).
-   **Visual Tests**: Use `ase.visualize` in UAT notebooks to inspect generated surfaces/defects.
-   **Strategy**: Deterministic seeds for random generation to ensure reproducible tests.

### CYCLE 03: Oracle
-   **Unit Tests**: Test the input file generation (e.g., `pw.x` input content).
-   **Mock Tests**: Use a `MockCalculator` that returns analytical forces (e.g., LJ) to test the `Oracle` class logic without running QE.
-   **Real Tests**: Run a tiny DFT calculation (e.g., Si bulk, 2 atoms) if `pw.x` is available.
-   **Strategy**: "Dry-run" modes to verify input syntax without expensive execution.

### CYCLE 04: Trainer
-   **Unit Tests**: Verify command-line argument construction for `pace_train`.
-   **Integration Tests**: Train on a small synthetic dataset (e.g., LJ data) and verify the potential file `.yace` is created.
-   **Strategy**: Check that the output potential predicts forces on the training set (sanity check).

### CYCLE 05: Dynamics Engine
-   **Unit Tests**: Verify `in.lammps` generation (correct pair_style, fix commands).
-   **Integration Tests**: Run a short MD with `fix halt`. Artificially inject high uncertainty (mock) to verify the Halt trigger works.
-   **Strategy**: Use "Soft" assertions for physical values (allows some fluctuation) but "Hard" assertions for execution flow (must halt on trigger).

### CYCLE 06: Validation & Orchestration
-   **End-to-End Tests**: Run the full "Fe/Pt on MgO" scenario (scaled down).
-   **Validation Tests**: Verify the `Validator` correctly flags bad potentials (e.g., imaginary phonons).
-   **UAT**: Execute the Tutorial Notebooks in CI mode.
-   **Strategy**: The "Final Exam". The system must survive a full cycle autonomously.
