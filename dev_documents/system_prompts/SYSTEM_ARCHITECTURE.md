# SYSTEM ARCHITECTURE

## 1. Summary

The PYACEMAKER project represents a paradigm shift in the development and operationalisation of Machine Learning Interatomic Potentials (MLIPs). At its core, the system leverages the Atomic Cluster Expansion (ACE) formalism, implemented via the "Pacemaker" engine, to generate potentials that combine the accuracy of first-principles methods with the computational efficiency of empirical force fields. The primary mission of this project is to democratize access to high-quality atomistic simulations. Historically, the creation of robust MLIPs has been the preserve of domain experts, requiring deep knowledge of Density Functional Theory (DFT), regression techniques, and sampling strategies. This manual, iterative process is often fraught with inefficiencies, such as the generation of redundant data and the risk of physical instabilities in extrapolation regimes.

PYACEMAKER addresses these challenges by introducing a fully automated, "Zero-Config" workflow. Users provide a single configuration file describing the material system and desired properties, and the system autonomously orchestrates the complex cycle of structure generation, labelling, training, and validation. This automation is achieved through a sophisticated Python-based architecture that integrates diverse computational tools—Quantum Espresso for DFT, LAMMPS for Molecular Dynamics (MD), and EON for Kinetic Monte Carlo (kMC)—into a cohesive pipeline.

A key innovation of the system is its focus on "Data Efficiency" and "Physical Robustness". By employing Active Learning techniques, specifically uncertainty-based sampling and D-optimality criteria, the system intelligently selects only the most informative atomic configurations for high-cost DFT calculations. This can reduce the computational burden by an order of magnitude compared to brute-force methods. Furthermore, the system enforces physical realism through the use of Hybrid Potentials, where a physics-based baseline (Lennard-Jones or ZBL) is augmented by the ACE many-body terms. This ensures that the potential behaves reasonably even in unexplored regions of the configuration space, preventing catastrophic simulation failures.

The system is designed to be modular and scalable. It consists of loosely coupled components—Generator, Oracle, Trainer, Dynamics, Validator—managed by a central Orchestrator. This design allows for seamless deployment across various computing environments, from local laptops to massive High-Performance Computing (HPC) clusters. It also facilitates future extensions, such as the integration of different DFT codes or advanced sampling algorithms. Ultimately, PYACEMAKER aims to accelerate materials discovery by providing researchers with a reliable, automated tool for generating "State-of-the-Art" potentials with minimal human effort.

## 2. System Design Objectives

The architectural decisions for PYACEMAKER are driven by a set of rigorous design objectives and constraints, ensuring the final system is robust, efficient, and user-friendly.

### 2.1. Zero-Config Automation
The system must be capable of executing a complete Active Learning workflow based solely on a structured configuration file (YAML). The goal is to eliminate the need for users to write ad-hoc Python scripts or shell scripts to glue different codes together. This "Zero-Config" philosophy ensures reproducibility and lowers the barrier to entry for non-experts. The system should handle all internal complexity, including file management, error recovery, and tool interfacing.

### 2.2. Maximised Data Efficiency
In computational materials science, the generation of ground-truth data (via DFT) is the most expensive step. Therefore, the system is designed to maximise the information gain per DFT calculation. We employ "Active Learning" strategies where the system autonomously identifies regions of the configuration space where the current potential is unreliable (high uncertainty). By selectively sampling these regions and using "Active Set" optimisation (based on D-optimality) to filter redundant structures, the system achieves high accuracy with a minimal training set size.

### 2.3. Physical Robustness and Stability
A common failure mode of MLIPs is "unphysical behaviour" in extrapolation regimes—such as atoms overlapping without penalty or crystals becoming unstable. To mitigate this, the system mandates the use of **Hybrid Potentials**. A physical baseline (LJ or ZBL) handles the core repulsion and long-range asymptotics, while the ACE model learns the complex many-body interactions. This ensures that simulations do not crash (e.g., due to segmentation faults from overlapping atoms) and that the potential remains physically sensible even outside the training domain.

### 2.4. Self-Healing and Resilience
The workflow involves complex external codes (Quantum Espresso, LAMMPS) that can fail for various numerical reasons (e.g., SCF convergence failure). The **Oracle** module is designed with self-healing capabilities. It detects failures and automatically attempts recovery strategies, such as mixing parameter reduction, smearing adjustment, or algorithm switching. This ensures that the pipeline is resilient and can run unattended for long periods without stalling due to minor numerical glitches.

### 2.5. Scalability and Extensibility
The architecture must support a wide range of operational scales, from quick tests on a local machine to massive production runs on HPC clusters. The modular design, interacting via abstract interfaces, allows individual components to be swapped or upgraded without affecting the rest of the system. For instance, the MD engine could be switched from LAMMPS to another code, or the DFT engine from Quantum Espresso to VASP, by simply implementing a new adapter class.

### 2.6. Rigorous Quality Assurance
Accuracy on a test set (RMSE) is necessary but not sufficient for a production-grade potential. The system includes a comprehensive **Validator** module that acts as a quality gate. It evaluates physical properties such as phonon dispersion stability, elastic constants, and equation of state (EOS) behaviour. Only potentials that pass these rigorous physical checks are promoted for deployment, ensuring that the output of PYACEMAKER is trustworthy for scientific research.

## 3. System Architecture

The PYACEMAKER system is architected as a modular, cycle-based application controlled by a central Orchestrator. The components interact through well-defined interfaces, exchanging domain objects (Structures, Potentials) rather than raw files where possible.

### 3.1. Core Components

1.  **Orchestrator**: The central controller. It manages the lifecycle of the active learning loop. It determines the current state, invokes the appropriate components (Generator, Oracle, Trainer, Dynamics, Validator), and manages the data flow between them. It is responsible for decision-making, such as when to stop exploration and start training, or when to terminate the loop based on convergence criteria.

2.  **Structure Generator (Explorer)**: This component is responsible for proposing candidate atomic structures. It employs an "Adaptive Exploration Policy" to intelligently sample the configuration space. Instead of random sampling, it uses heuristics (e.g., temperature ramping, defect introduction) and feedback from the current potential's uncertainty to target unexplored regions.

3.  **Oracle (Labeler)**: The interface to the ground truth (DFT). It manages the execution of Quantum Espresso calculations. It includes the logic for "Periodic Embedding"—carving out local clusters from large MD snapshots and embedding them in smaller periodic cells for efficient DFT calculation. It also handles the "Self-Healing" of DFT calculations.

4.  **Trainer (Learner)**: This component wraps the Pacemaker training engine. It manages the conversion of atomic structures into the specific formats required by Pacemaker (`.pckl.gzip`), configures the fitting hyperparameters (loss weights, regularisation), and executes the training process. It also implements "Active Set" selection to filter the training data.

5.  **Dynamics Engine (Sampler)**: The execution engine for MD and kMC simulations. It uses the current MLIP to run simulations and monitors the "Extrapolation Grade" ($\gamma$) in real-time. If the uncertainty exceeds a threshold, it halts the simulation and returns the problematic structure to the Orchestrator for labelling. This "On-the-Fly" (OTF) capability is crucial for the active learning loop.

6.  **Validator (QA)**: The quality assurance module. After each training cycle, it runs a suite of tests (Phonons, Elasticity, EOS) to verify the physical validity of the new potential. It provides a "Go/No-Go" decision for deployment.

### 3.2. Data Flow (The Active Learning Cycle)

The system operates in a cyclic manner:
1.  **Exploration**: The Dynamics Engine or Structure Generator explores the configuration space using the current potential.
2.  **Detection**: High-uncertainty structures are detected (via $\gamma$ monitoring) and extracted.
3.  **Selection**: The Orchestrator selects a subset of these structures (using D-optimality) to minimise redundancy.
4.  **Labelling**: The Oracle computes the true energy and forces for the selected structures (DFT).
5.  **Training**: The Trainer updates the potential using the new data, employing fine-tuning techniques.
6.  **Validation**: The Validator checks the new potential. If it passes, it is deployed for the next cycle.

### 3.3. Architecture Diagram

```mermaid
graph TD
    User[User] -->|Config (YAML)| Orchestrator
    Orchestrator -->|Control| Generator[Structure Generator]
    Orchestrator -->|Control| Oracle[Oracle / DFT Manager]
    Orchestrator -->|Control| Trainer[Trainer / Pacemaker]
    Orchestrator -->|Control| Dynamics[Dynamics Engine / LAMMPS]
    Orchestrator -->|Control| Validator[Validator]

    subgraph "Data Store"
        Dataset[(Dataset .pckl)]
        Potential[(Potential .yace)]
    end

    Generator -->|Candidate Structures| Oracle
    Dynamics -->|Uncertain Structures| Oracle
    Oracle -->|Labelled Structures| Dataset
    Dataset --> Trainer
    Trainer -->|New Potential| Potential
    Potential --> Dynamics
    Potential --> Validator
    Validator -->|Validation Report| Orchestrator
```

## 4. Design Architecture

The software design of PYACEMAKER emphasizes type safety, modularity, and clarity. It is built using modern Python standards (Python 3.12+) and leverages robust libraries like Pydantic for data validation and Typer for CLI management.

### 4.1. File Structure
The project follows a standard `src` layout with clear separation of concerns.

```ascii
src/
└── mlip_autopipec/
    ├── __init__.py
    ├── main.py                 # CLI Entry Point
    ├── config.py               # Global Configuration Models
    ├── factory.py              # Dependency Injection Factory
    ├── domain_models/          # Pydantic Models (Data)
    │   ├── structure.py
    │   ├── potential.py
    │   └── validation.py
    ├── interfaces/             # Abstract Base Classes
    │   ├── orchestrator.py
    │   ├── generator.py
    │   ├── oracle.py
    │   ├── trainer.py
    │   ├── dynamics.py
    │   └── validator.py
    ├── implementations/        # Concrete Implementations
    │   ├── simple_orchestrator.py
    │   ├── generator/
    │   ├── oracle/             # QE Interface
    │   ├── trainer/            # Pacemaker Interface
    │   ├── dynamics/           # LAMMPS Interface
    │   └── validator/
    └── utils/                  # Shared Utilities
        ├── logging.py
        └── embedding.py
```

### 4.2. Domain Models
The system uses Pydantic models to define the data structures passed between components.
-   **`Structure`**: Represents an atomic configuration. It encapsulates positions, cell parameters, atomic numbers, and optional properties (forces, energy, stress). It supports serialisation to/from JSON and ASE Atoms objects.
-   **`GlobalConfig`**: The root configuration object. It uses discriminated unions to allow polymorphic configuration of sub-components (e.g., selecting between `MockOracle` and `QuantumEspressoOracle`).
-   **`ValidationResult`**: Captures the outcome of validation tests, including pass/fail status, metrics (RMSE), and paths to generated artifacts (plots).

### 4.3. Interface-Based Design
All major components are defined by Abstract Base Classes (ABCs). This allows for:
-   **Mocking**: We can easily swap in `MockOracle` or `MockTrainer` for unit testing and rapid development cycles without needing external binaries.
-   **Extensibility**: Adding a new DFT code (e.g., VASP) only requires implementing the `BaseOracle` interface, without modifying the Orchestrator.

### 4.4. Key Considerations
-   **Statelessness**: Where possible, components are designed to be stateless or manage state explicitly via file persistence. The Orchestrator manages the workflow state.
-   **Error Handling**: The system uses a hierarchy of custom exceptions. Critical failures (e.g., DFT divergence) are caught and handled by specific recovery logic in the implementations, while unrecoverable errors bubble up to the Orchestrator for graceful shutdown.
-   **Logging**: A centralized logging system tracks the progress of the active learning loop, writing to both console and a structured log file for post-mortem analysis.

## 5. Implementation Plan

The project will be executed in 6 sequential cycles, following the AC-CDD methodology. Each cycle builds upon the previous one, adding specific functionality and verifying it through tests.

### Cycle 01: Core Infrastructure & Mocks
**Objective**: Establish the skeleton of the application.
-   Define the Abstract Base Classes (ABCs) for all components (`BaseOrchestrator`, `BaseGenerator`, `BaseOracle`, `BaseTrainer`, `BaseDynamics`, `BaseValidator`).
-   Implement `Mock` versions of all components. These mocks will simulate the behaviour of the real tools (e.g., `MockOracle` returning random forces) to allow testing the orchestration logic.
-   Implement the `GlobalConfig` Pydantic models and the YAML parsing logic.
-   Set up the CLI entry point (`main.py`) using Typer.
-   Set up the logging infrastructure.

### Cycle 02: Structure Generator
**Objective**: Implement the logic for creating atomic structures.
-   Implement the `StructureGenerator` concrete class.
-   Develop the "Adaptive Exploration Policy" logic (simplified for initial version, potentially rule-based).
-   Implement random structure generation and template-based generation (e.g., perturbing a bulk crystal).
-   Ensure structures are correctly converted to the internal `Structure` domain model.

### Cycle 03: Oracle (DFT Interface)
**Objective**: Connect to the ground truth provider.
-   Implement `DFTManager` (Concrete Oracle) using `ase.calculators.espresso`.
-   Implement the logic for generating Quantum Espresso input files (PWscf).
-   Implement the "Self-Healing" mechanism to handle convergence errors (adjusting mixing beta, smearing).
-   Implement "Periodic Embedding": the algorithm to carve out a cluster from a larger structure and place it in a periodic box for DFT calculation.

### Cycle 04: Trainer (Pacemaker Interface)
**Objective**: Enable potential fitting.
-   Implement `PacemakerTrainer`.
-   Develop the wrapper logic to call `pace_train` via subprocess.
-   Implement data conversion routines (ASE Atoms -> Pacemaker `.pckl.gzip`).
-   Implement the "Active Set" selection logic using `pace_activeset` to filter training data based on D-optimality.
-   Configure the Hybrid Potential setup (ACE + LJ/ZBL).

### Cycle 05: Dynamics Engine (MD & OTF)
**Objective**: Enable active exploration and uncertainty quantification.
-   Implement `DynamicsEngine` using `lammps` (Python interface or subprocess).
-   Implement the logic to construct `pair_style hybrid/overlay` commands for LAMMPS.
-   Implement the "On-the-Fly" (OTF) monitoring loop.
-   Configure the `fix halt` command in LAMMPS to stop simulation when the Extrapolation Grade ($\gamma$) exceeds a threshold.
-   Implement the extraction of high-uncertainty structures from MD dumps.

### Cycle 06: Orchestrator & Validation
**Objective**: Close the loop and ensure quality.
-   Implement the full logic of `SimpleOrchestrator`. It should loop through: Explore -> Detect -> Label -> Train -> Validate.
-   Implement the `Validator` concrete class.
-   Add Phonon calculation checks (using Phonopy interface).
-   Add Elastic Constant calculation checks.
-   Integrate everything into the final CLI command `mlip-pipeline run config.yaml`.

## 6. Test Strategy

Testing is integral to the development process. We will employ a tiered testing strategy to ensure correctness at the unit, integration, and system levels.

### 6.1. Unit Testing (Cycles 01-06)
-   **Scope**: Individual classes and functions.
-   **Approach**: Use `pytest`. Every module in `src` must have a corresponding test file in `tests`.
-   **Mocks**: Heavy use of the Mock components created in Cycle 01. For example, testing the Orchestrator logic using `MockOracle` and `MockTrainer` ensures the workflow logic is correct without running actual heavy calculations.
-   **Coverage**: Aim for high code coverage (>80%), particularly for the logic-heavy parts like the Orchestrator and Data Converters.

### 6.2. Integration Testing (Cycles 03-05)
-   **Scope**: Interaction between Python code and external tools (QE, Pacemaker, LAMMPS).
-   **Approach**:
    -   **Mock Binaries**: For tools that are hard to install in CI (like Quantum Espresso), we will create "Mock Binaries" (simple shell/Python scripts) that mimic the input/output behaviour of the real binaries. This allows us to test the input generation and output parsing logic.
    -   **Miniature Runs**: For tools that can be installed (e.g., LAMMPS, Pacemaker), we will run "Miniature" jobs (very small systems, few steps) to verify the interface works correctly.

### 6.3. System/End-to-End Testing (Cycle 06)
-   **Scope**: The entire `mlip-pipeline run` command.
-   **Approach**:
    -   **Mock Mode**: Run the full pipeline using the `Mock` components defined in Cycle 01. This verifies that the Orchestrator correctly transitions between states and handles data flow.
    -   **Real Mode (Smoke Test)**: If possible in the environment, run a very short real cycle (e.g., 1 iteration, small system) to verify the toolchain is intact.
    -   **Validation**: Check that the output files (potentials, logs, datasets) are created in the expected locations and have valid content.

### 6.4. Validation of Scientific Correctness
-   **Physics Checks**: The `Validator` module itself is a form of testing. We will have specific tests to ensure the `Validator` correctly identifies "bad" potentials (e.g., by feeding it a dummy potential that we know is unstable).
