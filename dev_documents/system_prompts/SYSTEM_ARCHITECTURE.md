# System Architecture: PyAcemaker

## 1. Summary

PyAcemaker is a cutting-edge, automated system designed to revolutionise the construction and operation of Machine Learning Interatomic Potentials (MLIPs). Built around the "Pacemaker" (Atomic Cluster Expansion) engine, this system addresses the significant barriers to entry in computational materials science by automating the complex workflow of potential generation. Traditionally, creating high-quality MLIPs required deep expertise in both data science and computational physics, involving manual iteration through structure generation, Density Functional Theory (DFT) calculations, training, and validation. This manual process is error-prone, time-consuming, and often leads to potentials that fail in "extrapolation" regions—configurations not represented in the training data.

PyAcemaker solves these problems by providing a "Zero-Config" workflow. A user need only provide a single YAML configuration file, and the system autonomously manages the entire lifecycle of the potential. It employs an Active Learning cycle that iteratively explores the chemical and structural space, detects regions of high uncertainty, performs targeted DFT calculations, and refines the potential. This approach ensures that the potential is not only accurate but also robust, capable of handling rare events and high-energy configurations without catastrophic failure. By integrating physics-informed baselines (such as Lennard-Jones or ZBL potentials) and advanced sampling techniques (like Adaptive Kinetic Monte Carlo), PyAcemaker ensures that the resulting potentials are both data-efficient and physically sound. The system is architected to be modular and scalable, suitable for deployment on local workstations or High-Performance Computing (HPC) clusters, making state-of-the-art MLIP construction accessible to a broader range of researchers and engineers.

## 2. System Design Objectives

The design of PyAcemaker is guided by several critical objectives, constraints, and success criteria, ensuring it meets the needs of modern materials research.

### Goals

1.  **Democratisation of MLIP Construction**: The primary goal is to lower the barrier to entry. Users with minimal programming experience should be able to generate state-of-the-art potentials. The system abstracts away the complexities of DFT parameter tuning, training hyperparameter optimisation, and validation protocols.
2.  **Zero-Config Automation**: The system must operate autonomously from a single input file. It should handle error recovery (e.g., DFT convergence failures), resource management, and workflow orchestration without human intervention.
3.  **Data Efficiency**: Unlike brute-force approaches that generate vast amounts of redundant data, PyAcemaker aims to achieve high accuracy (Energy RMSE < 1 meV/atom, Force RMSE < 0.05 eV/Å) with a minimal number of DFT calculations. This is achieved through active learning strategies that select only the most informative structures for training.
4.  **Physical Robustness**: A critical flaw in many ML potentials is non-physical behaviour in extrapolation regions (e.g., lack of core repulsion). This system must guarantee physical safety by enforcing physics-informed baselines (Delta Learning) and rigorous validation checks (Phonon stability, Elastic constants).
5.  **Scalability**: The architecture must support scaling from simple unit cells to complex supercells with defects, and from short MD runs to long-timescale kMC simulations.

### Constraints

*   **Computational Resources**: DFT calculations are expensive. The system must optimise resource usage by using "Periodic Embedding" to calculate forces only for relevant atoms in a local environment, rather than entire large supercells.
*   **Software Dependencies**: The system relies on external engines like Quantum Espresso, LAMMPS, and Pacemaker. It must interface with these tools robustly, handling version differences and execution environments (e.g., via Docker/Singularity).
*   **Security**: As an automated system executing external commands, it must be secured against injection attacks and ensure safe handling of file paths and process execution.

### Success Criteria

*   **Automation Level**: Complete execution of the active learning loop without manual intervention.
*   **Accuracy**: Achievement of target RMSE values on hold-out test sets.
*   **Stability**: No "segmentation faults" during MD simulations due to unphysical potential behaviour.
*   **Usability**: A clear, comprehensive dashboard that allows users to monitor the progress and quality of the potential generation in real-time.

## 3. System Architecture

The system is designed as a set of loosely coupled modules orchestrated by a central controller. This modularity ensures maintainability and allows for individual components to be upgraded or replaced without affecting the whole system.

### Components

1.  **Orchestrator**: The "Brain" of the system. It manages the active learning loop, tracks the state of the workflow, and coordinates data flow between other modules. It decides when to explore, when to train, and when to validate.
2.  **Structure Generator (Explorer)**: The "Explorer". It proposes new atomic configurations. Unlike simple random sampling, it uses an "Adaptive Exploration Policy" to intelligently sample the configuration space using MD, MC, or defect generation based on the material's properties.
3.  **Oracle**: The "Wise Man". It performs DFT calculations to label the proposed structures. It includes a "Self-Healing" mechanism to automatically fix convergence errors and a "Periodic Embedding" feature to efficient calculation of local forces.
4.  **Trainer**: The "Learner". It wraps the Pacemaker engine to train the ACE potential. It manages the dataset, applies Delta Learning with physical baselines, and optimises the "Active Set" of training data to prevent redundancy.
5.  **Dynamics Engine**: The "Executor". It runs MD or kMC simulations using the trained potential. It features an "Uncertainty Watchdog" that halts the simulation if the potential encounters a configuration with high uncertainty ($\gamma$ value), triggering a learning update.
6.  **Validator**: The "Gatekeeper". It performs rigorous physics-based tests (Phonons, Elasticity, EOS) to ensure the potential is not just numerically accurate but physically meaningful.

### Data Flow

The data flows in a cyclical manner:
1.  **Exploration**: The Dynamics Engine or Structure Generator creates new structures.
2.  **Detection**: High-uncertainty structures are identified.
3.  **Selection**: Representative structures are selected and prepared (Periodic Embedding).
4.  **Calculation**: The Oracle calculates energy and forces (Labels).
5.  **Refinement**: The Trainer updates the potential with the new data.
6.  **Deployment**: The new potential is deployed back to the Dynamics Engine.

### Diagram

```mermaid
graph TD
    User[User Configuration] --> Orch[Orchestrator]
    Orch --> SG[Structure Generator]
    Orch --> DE[Dynamics Engine<br/>(LAMMPS/EON)]
    Orch --> Oracle[Oracle<br/>(DFT/QE)]
    Orch --> Trainer[Trainer<br/>(Pacemaker)]
    Orch --> Valid[Validator]

    subgraph "Active Learning Loop"
        SG -->|Candidates| Oracle
        DE -->|Halted Structures| Oracle
        Oracle -->|Labelled Data| Trainer
        Trainer -->|Potential.yace| DE
        Trainer -->|Potential.yace| Valid
    end

    Valid -->|Pass/Fail| Orch
    DE -->|Uncertainty Metric| Orch
```

## 4. Design Architecture

The system is implemented in Python, leveraging strict type hinting (Pydantic) for configuration and data management. This ensures robustness and clarity in the codebase.

### File Structure

```ascii
src/
├── mlip_autopipec/
│   ├── app.py                  # Entry point (CLI)
│   ├── orchestrator.py         # Central control logic
│   ├── config/                 # Pydantic Configuration Models
│   │   ├── __init__.py
│   │   ├── main_config.py
│   │   └── module_configs.py
│   ├── generator/              # Structure Generation
│   │   ├── policy.py
│   │   └── defects.py
│   ├── dft/                    # Oracle (DFT)
│   │   ├── runner.py
│   │   ├── input_gen.py
│   │   └── embedding.py
│   ├── training/               # Trainer (Pacemaker)
│   │   ├── wrapper.py
│   │   └── dataset.py
│   ├── inference/              # Dynamics Engine
│   │   ├── lammps_runner.py
│   │   ├── eon_wrapper.py
│   │   └── watchdog.py
│   ├── validation/             # Validation Suite
│   │   ├── phonons.py
│   │   ├── elastic.py
│   │   └── eos.py
│   └── utils/
│       ├── logging_setup.py
│       └── helpers.py
```

### Data Models

The system relies heavily on Pydantic models to define interfaces.

*   **GlobalConfig**: The root configuration object, validated against the user's YAML input.
*   **StructureData**: A standard wrapper around ASE Atoms objects, including metadata about their origin (e.g., "exploration", "halted") and their calculated properties.
*   **PotentialState**: Tracks the versioning of potentials, their training metrics, and validation status.
*   **WorkflowStatus**: Manages the state of the active learning loop, including current iteration, active module, and error counts.

### Key Class Definitions

*   `Orchestrator`: Singleton class that initialises modules and runs the main loop.
*   `QERunner`: Manages Quantum Espresso execution, including input file generation and error handling/retries.
*   `PacemakerWrapper`: Encapsulates `pace_train` and `pace_activeset` commands, managing file I/O for these external tools.
*   `LammpsRunner`: Handles LAMMPS execution, specifically managing the `fix halt` logic and parsing log files for uncertainty data.

## 5. Implementation Plan

The development is divided into 8 sequential cycles, each building upon the previous one.

*   **CYCLE01: Architecture Skeleton & Configuration**
    *   Set up the project structure, logging, and error handling.
    *   Define all Pydantic configuration models (`GlobalConfig`, `DFTConfig`, `TrainingConfig`, etc.).
    *   Implement the basic `Orchestrator` class with a mock loop to verify flow control.
*   **CYCLE02: Oracle (DFT Automation)**
    *   Implement `QERunner` and `DFTManager`.
    *   Develop input generation logic (SSSP, k-spacing).
    *   Implement the "Self-Healing" logic for DFT convergence failures.
    *   Implement "Periodic Embedding" for cutting out local environments.
*   **CYCLE03: Trainer (Pacemaker Integration)**
    *   Implement `PacemakerWrapper`.
    *   Develop logic for Delta Learning (LJ/ZBL baseline setup).
    *   Implement dataset management and "Active Set" optimisation (`pace_activeset`).
*   **CYCLE04: Dynamics Engine (LAMMPS Inference)**
    *   Implement `LammpsRunner`.
    *   Develop the `pair_style hybrid/overlay` input generation.
    *   Implement the Uncertainty Watchdog (Log parsing for `fix halt`).
    *   Implement logic to extract halted structures.
*   **CYCLE05: Active Learning Strategy (Selection & DB)**
    *   Connect the Dynamics Engine and Oracle via the selection logic.
    *   Implement "Local D-Optimality Selection" (selecting best candidates from halted structures).
    *   Implement a lightweight database or file-system manager for tracking structures and their states.
*   **CYCLE06: Validation Suite**
    *   Implement `Validator` class.
    *   Develop `PhononValidator` (using Phonopy).
    *   Develop `ElasticValidator` (Born criteria).
    *   Develop `EOSValidator` (Birch-Murnaghan).
    *   Implement the "Gatekeeper" logic (Pass/Fail/Conditional).
*   **CYCLE07: Advanced Exploration (EON & Generator)**
    *   Implement `EONWrapper` for kMC support.
    *   Implement `StructureGenerator` with "Adaptive Exploration Policies".
    *   Implement defect and strain engineering logic.
*   **CYCLE08: Full Loop Integration & Reporting**
    *   Final integration of all modules into the `Orchestrator`.
    *   Implement the CLI entry point (`mlip-auto run`).
    *   Develop the Dashboard/Reporting module (HTML report generation).
    *   Final End-to-End testing and documentation.

## 6. Test Strategy

A rigorous testing strategy is essential for a system that automates scientific calculations.

### Unit Testing
*   **Scope**: Individual functions and classes (e.g., input file generators, parsers, config validators).
*   **Tools**: `pytest` with `pytest-cov`.
*   **Approach**: Mock external dependencies (Quantum Espresso, LAMMPS, Pacemaker) to test logic without running heavy calculations. Verify that Pydantic models correctly reject invalid configurations.

### Integration Testing
*   **Scope**: Interaction between modules (e.g., Orchestrator -> DFTManager, Trainer -> Dataset).
*   **Approach**: Use lightweight mocks for the physics engines but test the file I/O and data flow between the Python modules. Ensure that a "halted" structure from the mocked Dynamics Engine is correctly passed to the Oracle and then to the Trainer.

### End-to-End (E2E) Testing
*   **Scope**: The entire workflow from `config.yaml` to a trained potential.
*   **Approach**:
    *   **Dry Run**: A mode where all external commands are printed/logged but not executed, verifying the command construction.
    *   **Toy System**: Run the full loop on a very simple system (e.g., Silicon unit cell) with very loose convergence criteria to ensure the pipeline finishes in a reasonable time on the CI/CD environment or local developer machine.
    *   **Regression Tests**: Ensure that changes in the code do not degrade the performance (RMSE) on a standard dataset.

### Validation Tests (Scientific Correctness)
*   **Scope**: The physical validity of the output.
*   **Approach**: This is part of the system's runtime logic (Cycle 06) but also serves as a test of the system's effectiveness. We verify that the system creates potentials that respect fundamental physics (positive bulk modulus, stable phonons for stable structures).
