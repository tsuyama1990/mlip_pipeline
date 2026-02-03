# System Architecture

## 1. Summary

The PYACEMAKER project represents a paradigm shift in the computational materials science domain, specifically addressing the "democratisation" of Machine Learning Interatomic Potentials (MLIPs). Traditionally, the construction of high-fidelity MLIPs has been the exclusive preserve of experts possessing deep dual competency in quantum mechanics (Density Functional Theory - DFT) and data science. The manual workflow involves an iterative, labor-intensive cycle of structure generation, DFT calculation, potential fitting, and validation, often plagued by trial-and-error inefficiencies. PYACEMAKER aims to obliterate these barriers by providing a "Zero-Config" automated pipeline that orchestrates the entire lifecycle of an ACE (Atomic Cluster Expansion) potential.

At its core, PYACEMAKER is an intelligent orchestrator that manages the complex interplay between four critical subsystems: the Structure Generator (the "Explorer"), the Oracle (the "Sage" managing DFT), the Trainer (the "Learner" wrapping Pacemaker), and the Dynamics Engine (the "Executor" running MD/kMC). Unlike static workflows, PYACEMAKER employs an Active Learning strategy where the system autonomously identifies "unknown" regions of the chemical space—signalled by high extrapolation grades ($\gamma$)—and deliberately samples them. This ensures that the resulting potential is not just accurate near equilibrium but physically robust in far-from-equilibrium regimes, such as high-temperature melts, defect cores, and reaction transition states.

The system is architected to be modular and container-native, ensuring reproducibility across diverse computing environments, from local workstations to High-Performance Computing (HPC) clusters. A key innovation is the "Physics-Informed Robustness" mechanism. By enforcing a physical baseline (Lennard-Jones or ZBL) and learning only the many-body residual, the system guarantees that the potential behaves safely (i.e., no nuclear fusion or non-physical collapse) even in regions where training data is sparse. Furthermore, the integration of Adaptive Kinetic Monte Carlo (aKMC) via EON allows the system to bridge the timescale gap, learning from rare events that occur over seconds or hours, which are inaccessible to standard Molecular Dynamics.

Ultimately, PYACEMAKER empowers experimentalists and non-expert simulators to generate "State-of-the-Art" potentials with minimal input—often just a single YAML configuration file. By automating the drudgery of data curation and fitting, it frees researchers to focus on the scientific insights derived from the simulations, accelerating the discovery of new materials and the understanding of complex atomistic phenomena. The project serves as a foundational tool for the next generation of materials informatics, providing a robust, scalable, and self-correcting platform for atomic-scale modelling.

## 2. System Design Objectives

The design of PYACEMAKER is driven by a set of rigorous objectives and constraints, ensuring that the final product is not only functional but also superior to existing manual workflows in terms of efficiency, reliability, and usability.

### 2.1 Goals

1.  **Zero-Configuration Automation**: The primary goal is to minimize user intervention. The system must accept a high-level intent (e.g., "Make a potential for TiO2") and handle all intermediate steps—parameter selection for DFT, hyperparameter tuning for the ML model, and sampling strategies—autonomously.
2.  **Data Efficiency**: We aim to achieve "chemical accuracy" (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with a fraction of the computational cost associated with random sampling. By using Active Learning (D-Optimality and Uncertainty Quantification), the system should only perform expensive DFT calculations on structures that maximize information gain.
3.  **Physical Robustness**: A critical failure mode in ML potentials is unphysical behaviour in extrapolation regions. The system must guarantee stability. This is achieved by enforcing a hard physical baseline (ZBL/LJ) for core repulsion, ensuring that simulations never crash due to "holes" in the potential energy surface.
4.  **Timescale and Lengthscale Scalability**: The architecture must support the transition from small-scale active learning (hundreds of atoms) to large-scale production runs (millions of atoms). Furthermore, it must integrate kMC to span timescales from femtoseconds to seconds, capturing slow diffusive processes.
5.  **Self-Healing and Resilience**: The system must be robust against component failures. If a DFT calculation fails (e.g., SCF non-convergence), the Oracle module must automatically attempt remedial actions (mixing adjustments, smearing) without crashing the entire pipeline.

### 2.2 Constraints

1.  **Strict Modularity**: Components (Oracle, Trainer, etc.) must communicate via defined interfaces and file formats (XYZ, YAML), avoiding tight coupling. This allows for individual components to be upgraded or swapped (e.g., changing the DFT engine from QE to VASP) without refactoring the Orchestrator.
2.  **Reproducibility**: All random seeds, configuration snapshots, and software versions must be logged. A "Production Manifest" must accompany every released potential, allowing complete traceability of its genealogy.
3.  **Resource Awareness**: The system must operate within the limits of the available hardware. It should support "Mock Mode" for CI/CD testing on limited resources and "HPC Mode" for massive parallel execution via schedulers like SLURM (future scope, but architecture must permit it).
4.  **Safety First**: No simulation should ever produce a "Segmentation Fault" due to unphysical atomic overlaps. The Hybrid Potential strategy is a hard constraint for all MD execution.

### 2.3 Success Criteria

-   **Automation**: A user can run `pyacemaker run config.yaml` and receive a `production_potential.zip` after a set duration without manual intervention.
-   **Quality**: The final potential passes the "Validation Suite" (Phonons, Elastic Constants, EOS) with no "Red" flags.
-   **Stability**: An MD run at $2 \times T_{melt}$ does not crash or exhibit unphysical density.
-   **Efficiency**: The Active Learning loop converges to the target accuracy with fewer than 10 iterations of the Refinement cycle.

## 3. System Architecture

The system follows a "Hub-and-Spoke" architecture, where the **Orchestrator** acts as the central hub, coordinating the activities of four specialized modules (Spokes). This design ensures separation of concerns and facilitates independent testing and development of each module.

### 3.1 Components

1.  **Orchestrator (The Brain)**:
    -   **Responsibility**: Manages the workflow state machine. It decides when to transition from Exploration to Selection, when to trigger Retraining, and when to terminate the job.
    -   **Data**: Reads `config.yaml`, maintains the `loop_state.json`, and manages the directory structure for each iteration.

2.  **Structure Generator / Explorer (The Explorer)**:
    -   **Responsibility**: Generates candidate atomic structures. It uses Molecular Dynamics (LAMMPS) and Monte Carlo methods to explore the configuration space.
    -   **Key Feature**: Implements "Adaptive Exploration". If the system detects it is stuck, the Explorer switches strategies (e.g., from MD to High-Temperature MC or Strain scanning).

3.  **Oracle (The Sage)**:
    -   **Responsibility**: Provides the "Ground Truth". It wraps the DFT code (Quantum Espresso).
    -   **Key Feature**: "Periodic Embedding". It takes a local cluster cut from a large MD simulation and embeds it into a periodic supercell suitable for DFT, ensuring valid boundary conditions. It also handles DFT error recovery.

4.  **Trainer (The Learner)**:
    -   **Responsibility**: Fits the ACE potential. It wraps the `pacemaker` suite.
    -   **Key Feature**: "Active Set Selection". It uses linear algebra (MaxVol algorithm) to select the most informative structures from the candidate pool, preventing data explosion.

5.  **Dynamics Engine (The Executor)**:
    -   **Responsibility**: Runs the simulations for application and validation.
    -   **Key Feature**: "On-the-Fly (OTF) Monitoring". It runs MD with a "Watchdog" that halts the simulation if the extrapolation grade $\gamma$ exceeds a safety threshold.

6.  **Validator (The Gatekeeper)**:
    -   **Responsibility**: Performs independent physics checks (Phonons, Elasticity).
    -   **Key Feature**: Blocks the release of a potential if it violates fundamental physical laws (e.g., imaginary phonons).

### 3.2 Mermaid Diagram

```mermaid
graph TD
    subgraph User Space
        Config[config.yaml] --> Orch[Orchestrator]
    end

    subgraph Core System
        Orch -->|1. Request Structures| Explorer[Structure Generator]
        Explorer -->|2. Candidate Structures| Orch

        Orch -->|3. Filter & Request Energy| Oracle[Oracle (DFT)]
        Oracle -->|4. Labelled Data (E, F, V)| Orch

        Orch -->|5. Update Dataset| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. New Potential (.yace)| Orch

        Orch -->|7. Run Validation| Validator[Validator]
        Validator -->|8. Report & Pass/Fail| Orch
    end

    subgraph External Engines
        Explorer -.-> LAMMPS[LAMMPS (MD)]
        Explorer -.-> EON[EON (aKMC)]
        Oracle -.-> QE[Quantum Espresso]
        Trainer -.-> PACE[Pacemaker / TensorFlow]
        Validator -.-> Phonopy[Phonopy]
    end

    subgraph "Active Learning Loop (OTF)"
        Dynamics[Dynamics Engine] -->|Monitor Gamma| Watchdog{High Uncertainty?}
        Watchdog -- Yes --> Halt[Halt & Extract]
        Halt -->|Local Structure| Explorer
        Watchdog -- No --> Continue[Continue Sim]
        Orch -->|Deploy Potential| Dynamics
    end
```

## 4. Design Architecture

The codebase is structured to enforce strict type checking (Pydantic), clear interfaces (Abstract Base Classes), and clean separation of data and logic.

### 4.1 File Structure

```
.
├── config.yaml                 # User configuration
├── pyproject.toml              # Project dependencies and tool config
├── README.md                   # Project documentation
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py             # Entry point
│       ├── config/             # Configuration definitions
│       │   ├── __init__.py
│       │   └── config_model.py # Pydantic models for config
│       ├── domain_models/      # Domain entities (Data Classes)
│       │   ├── __init__.py
│       │   ├── structures.py   # Structure metadata
│       │   ├── validation.py   # Validation results
│       │   └── dynamics.py     # MD/kMC states
│       ├── interfaces/         # Abstract Base Classes (Protocols)
│       │   ├── __init__.py
│       │   └── core_interfaces.py
│       ├── orchestration/      # Workflow logic
│       │   ├── __init__.py
│       │   ├── orchestrator.py # Main loop
│       │   └── otf_loop.py     # On-the-Fly logic
│       ├── physics/            # Scientific logic
│       │   ├── __init__.py
│       │   ├── oracle/
│       │   │   ├── manager.py  # DFT Manager
│       │   │   └── qe_parser.py
│       │   ├── structure_gen/
│       │   │   ├── explorer.py
│       │   │   └── strategies.py
│       │   ├── dynamics/
│       │   │   ├── lammps_runner.py
│       │   │   └── eon_wrapper.py
│       │   └── trainer/
│       │       └── pacemaker.py
│       ├── validation/         # Validation logic
│       │   ├── __init__.py
│       │   ├── runner.py
│       │   └── metrics.py      # Phonons, Elasticity
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── file_ops.py
├── tests/
│   ├── unit/
│   └── integration/
├── tutorials/
└── dev_documents/
```

### 4.2 Data Models

We utilize **Pydantic** for all data exchange to ensure data integrity.

-   **`SimulationConfig`**: The root configuration object, validated at startup.
-   **`StructureMetadata`**: Tracks the genealogy of every atom configuration (source, generation method, parent structure ID).
-   **`ValidationResult`**: A structured report containing pass/fail status and quantitative metrics for each validation test.
-   **`MDState`**: Represents the snapshot of a simulation, including current step, temperature, and any detected "Halt" events.

### 4.3 Key Interactions

1.  **Dependency Injection**: The `Orchestrator` does not instantiate concrete classes (like `QuantumEspressoOracle`) directly. Instead, it receives objects adhering to the `Oracle` protocol. This allows for easy mocking in tests.
2.  **State Persistence**: The workflow state is persisted to `active_learning/state.json` after every major step. This allows the system to resume from a crash without losing progress.
3.  **Error Propagation**: Custom exception classes (e.g., `DFTConvergenceError`, `PotentialStabilityError`) are used to handle domain-specific failures gracefully.

## 5. Implementation Plan

The development is divided into 6 sequential cycles, each delivering a functional increment of the system.

### Cycle 01: Core Framework & Infrastructure

**Objective**: Establish the skeleton of the application, configuration management, and the logging infrastructure.
**Features**:
-   **Project Setup**: Initialize `pyproject.toml`, directory structure, and Git repository.
-   **Configuration Engine**: Implement `config_model.py` using Pydantic to parse and validate `config.yaml`. This defines the schema for all future modules.
-   **Logging System**: Setup a structured logger that outputs to both console and file, critical for debugging long-running scientific jobs.
-   **Abstract Interfaces**: Define the Python Protocols (`Explorer`, `Oracle`, `Trainer`, `Validator`) in `core_interfaces.py`.
-   **Orchestrator Skeleton**: Implement the basic state machine of the `Orchestrator` class that can load config and instantiate (mock) components.
-   **CLI Entry Point**: Create `main.py` with `Typer` or `argparse` to run the application.

### Cycle 02: The Oracle (DFT Automation)

**Objective**: Implement the Data Generation capability (The "Sage").
**Features**:
-   **DFT Manager**: Implement `physics/oracle/manager.py` to handle the queueing and execution of DFT jobs.
-   **QE Interface**: Implement `physics/oracle/qe_parser.py` to generate `pw.x` input files and parse XML/text output.
-   **Periodic Embedding**: Implement the geometry manipulation logic to take a cluster of atoms and place them into a periodic box with appropriate padding.
-   **Error Handling**: Implement `SelfCorrection` logic for DFT (e.g., if SCF fails, reduce mixing beta and retry).
-   **Mock Oracle**: A robust mock implementation that returns synthetic DFT data (using a simple LJ potential or pre-calculated data) for testing without QE installed.

### Cycle 03: The Structure Generator (Exploration Phase 1)

**Objective**: Implement the Strategy to sample the chemical space (The "Explorer").
**Features**:
-   **Base Explorer**: Implement `physics/structure_gen/explorer.py`.
-   **Strategies**: Implement `physics/structure_gen/strategies.py` containing:
    -   `RandomDisplacement`: Jiggle atoms.
    -   `StrainScan`: Apply deformations.
    -   `HighTempMD`: Basic MD implementation using ASE or LAMMPS wrapper.
-   **ASE Integration**: Ensure all structure manipulations use `ase.Atoms` objects efficiently.
-   **Data Persistence**: Logic to save generated structures to disk (XYZ/ExtXYZ format) with metadata.

### Cycle 04: The Trainer (Pacemaker Integration)

**Objective**: Implement the Machine Learning core (The "Learner").
**Features**:
-   **Pacemaker Wrapper**: Implement `physics/trainer/pacemaker.py` to call `pace_train`, `pace_collect`, etc., via `subprocess`.
-   **Active Set Selection**: Implement the logic to filter the dataset using D-Optimality (calling `pace_activeset`).
-   **Hybrid Potential Setup**: Logic to define the reference potential (LJ/ZBL) and configure Pacemaker to learn the delta.
-   **Basic Validation**: Implement the RMSE check (Energy/Force/Stress) against the test set.

### Cycle 05: The Active Learning Loop (OTF Dynamics)

**Objective**: Close the loop with On-the-Fly active learning (The "Executor").
**Features**:
-   **Dynamics Engine**: Implement `physics/dynamics/lammps_runner.py` capable of running MD.
-   **Watchdog Integration**: Implement the `fix halt` logic in LAMMPS input generation to stop simulation on high gamma.
-   **Halt & Diagnose**: The Orchestrator logic to catch a Halt event, identify the problematic atom/cluster, extract it, and send it to the Oracle.
-   **Loop Logic**: Connect Explorer -> Oracle -> Trainer -> Dynamics -> Explorer.
-   **Uncertainty Quantification**: Parsing `gamma` values from MD logs.

### Cycle 06: Advanced Physics & Deployment

**Objective**: Production readiness, advanced physics (kMC), and user tutorials.
**Features**:
-   **EON Integration**: Implement `physics/dynamics/eon_wrapper.py` and the `pace_driver.py` script to allow EON to use the ACE potential.
-   **Advanced Validation**: Implement `physics/validation/metrics.py` for Phonons (Phonopy integration) and Elastic Constants.
-   **Reporting**: Generate HTML reports summarizing the validation results.
-   **Packaging**: Create the `ProductionDeployer` to zip the final potential with the manifest.
-   **Tutorials**: Create the Jupyter Notebooks for the Fe/Pt on MgO scenario.
-   **Final Polish**: Ensure all tests pass and documentation is complete.

## 6. Test Strategy

Testing a scientific workflow requires a multi-layered approach, distinguishing between code correctness and scientific validity.

### Cycle 01 Testing
-   **Unit Tests**: Verify Pydantic models reject invalid configs. Test Logging output formats.
-   **Integration Tests**: Verify the Orchestrator can "run" a cycle with Mock components without crashing.

### Cycle 02 Testing
-   **Unit Tests**: Test the input file generator for QE (verify flags). Test the parser against sample QE output files (success and failure cases).
-   **Integration Tests**: Run the Mock Oracle to ensure it returns data in the expected ASE format.

### Cycle 03 Testing
-   **Unit Tests**: Verify `RandomDisplacement` moves atoms correctly. Verify `StrainScan` changes cell vectors.
-   **Integration Tests**: Test the `Explorer` generating a sequence of structures and saving them to files.

### Cycle 04 Testing
-   **Unit Tests**: Mock the `subprocess` calls to Pacemaker to verify command-line arguments are constructed correctly.
-   **Integration Tests**: Run a "dummy" training on a small dataset (using a tiny model config) to verify the file pipeline works.

### Cycle 05 Testing
-   **Unit Tests**: Verify the LAMMPS input script generation includes the `fix halt` command.
-   **Integration Tests**: The "Skeleton Loop". Run the full cycle using Mocks. Mock the "Halt" event (force the Mock MD to return a halt status) and verify the Orchestrator triggers the extraction and retraining logic.

### Cycle 06 Testing
-   **System Tests**: The "End-to-End" run. Execute the `01_quickstart.ipynb` tutorial in CI (Mock Mode).
-   **Validation Tests**: Run Phonopy on a known simple structure (e.g., Silicon) using a known good potential to verify the interface works.
-   **User Acceptance**: Verify the generated `zip` file contains all necessary artifacts.
