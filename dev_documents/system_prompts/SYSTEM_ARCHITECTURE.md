# PYACEMAKER System Architecture

## 1. Summary

The PYACEMAKER project represents a paradigm shift in the construction and operation of Machine Learning Interatomic Potentials (MLIPs). Traditionally, the development of high-fidelity potentials has been the domain of experts, requiring intricate knowledge of Density Functional Theory (DFT), molecular dynamics (MD), and machine learning regression techniques. This manual process is often fraught with inefficiencies, such as the generation of redundant data, the risk of unphysical extrapolation in rare events, and the significant computational cost associated with trial-and-error parameter tuning.

PYACEMAKER addresses these challenges by providing a fully automated, zero-configuration workflow centered around the "Pacemaker" (Atomic Cluster Expansion) engine. The system is designed to democratize MLIP creation, allowing material scientists to input a simple configuration file (YAML) and receive a production-ready potential. By automating the feedback loop between exploration (MD/MC), labeling (DFT), and training (ACE), the system ensures that the potential is not only accurate but also physically robust and data-efficient.

The core philosophy of PYACEMAKER is "Active Learning." Instead of training on a static dataset, the system dynamically explores the chemical and structural space. It employs an intelligent "Orchestrator" that coordinates various agents: a "Structure Generator" that proposes candidate structures based on adaptive policies; an "Oracle" that performs self-healing DFT calculations to generate ground-truth labels; a "Trainer" that optimizes the ACE potential; and a "Dynamics Engine" that runs simulations while monitoring uncertainty. When the simulation encounters an unknown region (high uncertainty), the system halts, diagnoses the configuration, and triggers a targeted learning cycle. This "Halt & Diagnose" mechanism prevents the catastrophic failures common in standard MLIP workflows.

Furthermore, PYACEMAKER bridges the gap between different time scales. It integrates Molecular Dynamics (MD) for capturing fast thermal vibrations and diffusion with Adaptive Kinetic Monte Carlo (aKMC) for simulating long-term evolution such as ordering and phase transitions. This multi-scale approach enables the simulation of complex phenomena like hetero-epitaxial growth and interface formation, as demonstrated in the primary use case of Fe/Pt deposition on MgO substrates. The system is built with modularity and scalability in mind, utilizing containerization (Docker/Singularity) to ensure reproducibility and ease of deployment across diverse computing environments, from local workstations to High-Performance Computing (HPC) clusters.

## 2. System Design Objectives

The design of the PYACEMAKER system is guided by a set of rigorous objectives, constraints, and success criteria, ensuring that it meets the high standards required for scientific research and industrial application.

### Goals

1.  **Zero-Configuration Workflow**: The primary goal is to minimize human intervention. A user should be able to initiate a complex active learning campaign with a single command and a concise YAML configuration file. The system must handle all intermediate steps, including error handling in DFT calculations, hyperparameter tuning for training, and decision-making during exploration.
2.  **Data Efficiency**: We aim to achieve state-of-the-art accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with a fraction of the data required by traditional random sampling methods. By utilizing D-Optimality and uncertainty-driven sampling, the system ensures that every DFT calculation contributes maximally to the potential's improvement.
3.  **Physical Robustness**: The potential must be safe to use. It should never output unphysical forces that cause simulations to explode, even in extrapolation regions. This is achieved through "Delta Learning," where the ACE potential learns the residual errors of a physics-based baseline (LJ/ZBL), ensuring correct behavior at short interatomic distances.
4.  **Scalability**: The architecture must support the transition from small-scale testing to massive production runs. This includes handling large datasets, parallelizing DFT calculations, and supporting long-timescale simulations via aKMC.

### Constraints

*   **Computational Resources**: The system must be mindful of computational costs. DFT calculations are expensive, so the "Oracle" must be judicious in its execution, avoiding redundant calculations.
*   **Time-Scale Bridge**: The system must effectively link MD (femtoseconds to nanoseconds) and kMC (seconds to hours), requiring seamless data exchange and consistent potential definitions between different simulation engines (LAMMPS and EON).
*   **Dependency Management**: The system relies on external engines (Quantum Espresso, LAMMPS, Pacemaker, EON). It must manage these dependencies robustly, providing clear feedback if tools are missing or incompatible.

### Success Criteria

*   **Automation Level**: Successful completion of the "Fe/Pt on MgO" deposition and ordering scenario without manual intervention after the initial setup.
*   ** robustness**: Zero "Segmentation Faults" or simulation explosions during the exploration phase, thanks to the hybrid potential strategy and the uncertainty watchdog.
*   **Accuracy**: Validation metrics (phonon dispersion, elastic constants, EOS) falling within 10-15% of DFT or experimental reference values.
*   **Usability**: New users should be able to run the provided "Jupyter Notebook" tutorials and reproduce key results within a reasonable timeframe (using mock modes for quick verification).

## 3. System Architecture

The PYACEMAKER system is architected as a modular, event-driven pipeline orchestrated by a central Python controller. The components are designed to be loosely coupled, communicating through well-defined interfaces and data structures, which facilitates testing and future extensions.

### High-Level Components

1.  **Orchestrator**: The brain of the operation. It manages the active learning loop, deciding when to explore, when to train, and when to validate. It maintains the state of the project and handles file management and logging.
2.  **Structure Generator**: The explorer. It generates atomic structures for training. Instead of random generation, it uses an "Adaptive Exploration Policy" to intelligently sample relevant regions of the chemical space, such as surfaces, interfaces, and defects, based on the current uncertainty landscape.
3.  **Oracle**: The ground truth provider. It interfaces with DFT codes (Quantum Espresso, VASP) to calculate energies, forces, and stresses. It includes a "Self-Healing" mechanism to automatically recover from common SCF convergence errors.
4.  **Trainer**: The learner. It wraps the Pacemaker engine to fit the ACE potential. It manages the training dataset, performs "Active Set Selection" to filter redundant data, and executes the fitting process with physics-informed regularization.
5.  **Dynamics Engine**: The simulator. It runs MD (via LAMMPS) and aKMC (via EON) simulations. Crucially, it implements the "On-the-Fly (OTF)" monitoring system that halts simulations when extrapolation uncertainty exceeds a safety threshold.
6.  **Validator**: The auditor. It runs a suite of physical tests (Phonons, Elasticity, EOS) on the generated potential to ensure it is not just numerically accurate but physically meaningful.

### Data Flow

The data flows in a cyclical manner, driven by the "Active Learning Cycle":
1.  **Exploration**: The Dynamics Engine runs simulations using the current potential.
2.  **Detection**: The engine detects high uncertainty ($\gamma > \text{threshold}$) and halts.
3.  **Selection**: The Orchestrator extracts the problematic local environment and the Structure Generator creates diverse local candidates.
4.  **Calculation**: The Oracle performs DFT calculations on selected candidates (embedded in periodic cells).
5.  **Refinement**: The Trainer updates the potential using the new data.
6.  **Deployment**: The new potential is hot-swapped into the Dynamics Engine, and the simulation resumes.

### Mermaid Diagram

```mermaid
graph TD
    subgraph Control Plane
        Orchestrator[Orchestrator]
        Config[Configuration (YAML)]
    end

    subgraph "Exploration & Dynamics"
        Gen[Structure Generator]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Watchdog{Uncertainty Watchdog}
    end

    subgraph "Learning & Verification"
        Oracle[Oracle (DFT/QE)]
        Trainer[Trainer (Pacemaker)]
        Validator[Validator]
    end

    Config --> Orchestrator
    Orchestrator --> Gen
    Orchestrator --> Dyn
    Orchestrator --> Oracle
    Orchestrator --> Trainer
    Orchestrator --> Validator

    Gen -- "Candidate Structures" --> Oracle
    Dyn -- "Halted Structure" --> Gen
    Dyn -- "Trajectory" --> Watchdog
    Watchdog -- "High Uncertainty" --> Orchestrator

    Oracle -- "Labeled Data (E, F, S)" --> Trainer
    Trainer -- "Potential (.yace)" --> Dyn
    Trainer -- "Potential (.yace)" --> Validator
    Validator -- "Quality Report" --> Orchestrator
```

## 4. Design Architecture

The internal design of the system relies on strict typing and robust data models to ensure reliability and maintainability. We utilize Python's `pydantic` library for configuration management and data validation, and `ASE` (Atomic Simulation Environment) for handling atomic structures.

### File Structure

```
.
├── config.yaml               # User configuration
├── pyproject.toml            # Project dependencies and tool config
├── README.md                 # Project documentation
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py           # Entry point
│       ├── core/
│       │   ├── orchestrator.py
│       │   ├── config.py     # Pydantic models
│       │   └── logger.py
│       ├── domain_models/    # Data Transfer Objects
│       │   ├── inputs.py
│       │   ├── results.py
│       │   └── enums.py
│       ├── components/       # Interface definitions & Implementations
│       │   ├── base.py
│       │   ├── generator.py
│       │   ├── oracle.py
│       │   ├── trainer.py
│       │   ├── dynamics.py
│       │   └── validator.py
│       ├── utils/
│       │   ├── lammps_driver.py
│       │   ├── eon_driver.py
│       │   └── file_manager.py
│       └── constants.py
├── tests/                    # Unit and Integration tests
└── dev_documents/            # Documentation
```

### Class & Data Definitions

*   **Pydantic Models**: All configurations are defined as Pydantic models (e.g., `OrchestratorConfig`, `DFTConfig`). This ensures that invalid user inputs are caught early with descriptive error messages.
*   **Abstract Base Classes (ABCs)**: Each component (Generator, Oracle, etc.) inherits from an ABC defined in `components/base.py`. This enforces a consistent interface (e.g., `generate()`, `compute()`, `train()`) and allows for easy swapping of implementations (e.g., swapping `QuantumEspressoOracle` for `VaspOracle` or `MockOracle`).
*   **Domain Models**: We define standard objects for passing data between components. For example, a `CandidateStructure` object might encapsulate the `ASE.Atoms` object along with metadata about its provenance (e.g., "generated by random displacement from halt #5").
*   **Dependency Injection**: The Orchestrator instantiates components based on the configuration. This allows for a "Mock Mode" where real heavy-lifting components are replaced by mocks for fast testing and CI/CD pipelines.

## 5. Implementation Plan

The project will be executed in 8 sequential cycles. Each cycle builds upon the previous one, adding complexity and functionality in a controlled manner.

### Cycle 01: Core Framework & Orchestrator Skeleton
*   **Objective**: Establish the project foundation.
*   **Features**:
    *   Set up the directory structure and `pyproject.toml`.
    *   Implement the `Orchestrator` class with the main event loop skeleton.
    *   Define Pydantic configuration models (`config.py`).
    *   Implement the logging system.
    *   Create "Mock" versions of all major components (`MockGenerator`, `MockOracle`, etc.) to verify the data flow without running external heavy codes.
    *   Implement the CLI entry point (`main.py`).

### Cycle 02: Structure Generator & Adaptive Policy
*   **Objective**: Implement intelligent structure creation.
*   **Features**:
    *   Develop the `StructureGenerator` class.
    *   Implement the `AdaptiveExplorationPolicy` logic that decides sampling strategies (MD vs MC, Temperature, Strain) based on input features.
    *   Implement specific sampling methods: Random Displacement, Supercell creation, Defect introduction.
    *   Integrate `M3GNet` (optional/mocked) for "Cold Start" initial exploration.

### Cycle 03: Oracle & DFT Automation
*   **Objective**: Enable automated, robust ground-truth generation.
*   **Features**:
    *   Implement `DFTOracle` with a focus on Quantum Espresso (`Espresso` calculator in ASE).
    *   Implement the "Self-Healing" logic to handle SCF convergence failures (adjusting mixing beta, smearing, etc.).
    *   Implement "Periodic Embedding": logic to cut out a local cluster from a large halted structure and wrap it in a periodic box for DFT.
    *   Ensure proper handling of pseudopotentials (SSSP) and k-point grid generation.

### Cycle 04: Trainer & Active Learning
*   **Objective**: Automate the training of ACE potentials.
*   **Features**:
    *   Implement `PacemakerTrainer` to wrap the `pace_train` command.
    *   Implement "Active Set Selection" using `pace_activeset` (D-Optimality) to filter training data.
    *   Implement database management: updating the `.pckl.gzip` dataset with new DFT results.
    *   Implement "Delta Learning" setup: configuring the trainer to learn the difference between DFT and a reference potential (LJ/ZBL).

### Cycle 05: Dynamics Engine (MD) & Hybrid Potential
*   **Objective**: Enable safe molecular dynamics simulations.
*   **Features**:
    *   Implement `LAMMPSDynamics` using the `lammps` Python interface or `subprocess`.
    *   Implement the `pair_style hybrid/overlay` logic to combine ACE with ZBL/LJ for safety.
    *   Implement the "Uncertainty Watchdog": configuring `compute pace` and `fix halt` in LAMMPS to stop simulations when $\gamma$ is high.
    *   Implement log parsing to detect halt conditions and extract the problematic snapshot.

### Cycle 06: The OTF Loop Integration
*   **Objective**: Close the loop between Dynamics, Oracle, and Trainer.
*   **Features**:
    *   Integrate the "Halt & Diagnose" workflow into the Orchestrator.
    *   Connect the output of `LAMMPSDynamics` (halted structure) to `StructureGenerator` (local candidates).
    *   Pass candidates to `Oracle` for labeling.
    *   Pass labels to `Trainer` for refinement.
    *   Implement the "Resume" logic: restarting LAMMPS with the new potential.

### Cycle 07: Advanced Dynamics (aKMC & Deposition)
*   **Objective**: Extend capabilities to long timescales and complex scenarios.
*   **Features**:
    *   Implement `EONWrapper` to interface with the EON software for aKMC.
    *   Develop the python driver script for EON to call the ACE potential.
    *   Implement logic for the "Fe/Pt Deposition" scenario: setup `fix deposit` in LAMMPS.
    *   Implement logic for bridging the final MD state to the initial aKMC state.

### Cycle 08: Validator & Quality Assurance
*   **Objective**: Ensure the scientific validity of the generated potential.
*   **Features**:
    *   Implement `StandardValidator` to run physical tests.
    *   Implement Phonon stability checks (via `Phonopy` wrapper).
    *   Implement Elastic constant calculation (Stress-Strain method).
    *   Implement Equation of State (EOS) fitting (Birch-Murnaghan).
    *   Generate the `validation_report.html` summarizing all metrics and plots.

## 6. Test Strategy

Testing is a critical part of the development process to ensure reliability and scientific accuracy. We will employ a multi-layered testing strategy.

### Cycle 01 Testing
*   **Unit Tests**: Verify config parsing and validation. Test `Orchestrator` state transitions.
*   **Integration Tests**: Run the full pipeline with "Mock" components. Verify that the loop completes, logs are generated, and "mock" data flows correctly from Generator to Trainer.

### Cycle 02 Testing
*   **Unit Tests**: Test each sampling method (random displacement, strain). Verify `AdaptiveExplorationPolicy` returns correct parameters for given inputs.
*   **Integration Tests**: Generate a batch of structures and verify they satisfy geometric constraints (e.g., minimum atomic distance).

### Cycle 03 Testing
*   **Unit Tests**: Test `DFTOracle` input file generation. Test the "Self-Healing" logic by simulating SCF failures (mocking the calculator to raise errors).
*   **Integration Tests**: Run a real (small) DFT calculation if Quantum Espresso is available (or use a high-fidelity mock that returns pre-calculated energies). Verify parsing of output forces and stresses.

### Cycle 04 Testing
*   **Unit Tests**: Test `PacemakerTrainer` command construction. Test Active Set selection logic (mocking `pace_activeset` output).
*   **Integration Tests**: Perform a training run on a small dummy dataset. Verify that a valid `.yace` file is produced.

### Cycle 05 Testing
*   **Unit Tests**: Test `LAMMPSDynamics` input script generation. Verify `pair_style hybrid` strings.
*   **Integration Tests**: Run a short LAMMPS MD with a dummy potential. Verify that `fix halt` triggers correctly when a threshold variable is manually manipulated (or using a mock potential that reports high gamma).

### Cycle 06 Testing
*   **System Tests**: Run the full "Closed Loop" with a simple system (e.g., Lennard-Jones Argon disguised as an ACE problem).
*   **Verification**: Ensure that a halt triggers a retraining cycle, and the subsequent run lasts longer (or the potential energy decreases).

### Cycle 07 Testing
*   **Unit Tests**: Test `EONWrapper` config generation.
*   **Integration Tests**: Run a short EON kMC step using a mock potential driver. Test the "Deposition" script in LAMMPS to ensure atoms are inserted correctly.

### Cycle 08 Testing
*   **Unit Tests**: Test the calculation of bulk modulus and shear modulus from stress-strain data.
*   **Integration Tests**: Run the full validation suite on a known good potential (e.g., a standard EAM potential wrapped as ACE). Verify that the report correctly identifies it as "Stable".
