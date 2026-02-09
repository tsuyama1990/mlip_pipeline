# System Architecture: PyAceMaker

## 1. Summary

The PyAceMaker project represents a paradigm shift in the construction and operation of Machine Learning Interatomic Potentials (MLIP). At its core, it leverages the "Pacemaker" tool, which utilizes Atomic Cluster Expansion (ACE), to provide a robust framework for material simulations. The primary objective of this system is to democratise the creation of "State-of-the-Art" MLIPs, enabling researchers with limited expertise in data science or computational physics to generate high-fidelity potentials with minimal manual intervention.

In the contemporary landscape of computational materials science, MLIPs serve as a crucial bridge between the high accuracy of Density Functional Theory (DFT) and the large-scale capabilities of classical Molecular Dynamics (MD). However, the traditional workflow for constructing these potentials is fraught with challenges. It often requires a meticulous, iterative process of manual structure generation, DFT calculation, training, and validation. This process is not only time-consuming but also prone to human error and inefficiency. Specifically, standard equilibrium MD simulations frequently fail to sample "rare events" or high-energy configurations, leading to potentials that behave unpredictably in unknown regions—the so-called extrapolation problem. Furthermore, the accumulation of redundant data (structures that are physically similar) wastes valuable computational resources without improving the potential's accuracy.

PyAceMaker addresses these issues through a fully automated, "Zero-Config" workflow. By employing an Orchestrator-based architecture, the system autonomously manages the entire lifecycle of the potential. From the initial generation of structures using adaptive exploration policies to the rigorous validation of the final model, every step is governed by intelligent algorithms. The system features an "Active Learning" loop that dynamically identifies regions of high uncertainty in the potential energy surface. When the system encounters a configuration where the potential's prediction is unreliable (high extrapolation grade), it automatically halts the simulation, extracts the problematic structure, performs a targeted DFT calculation (Oracle), and retrains the model. This self-healing capability ensures that the potential becomes progressively more robust and accurate over time.

Moreover, the system incorporates physics-informed constraints, such as hybrid potentials (combining ACE with Lennard-Jones or ZBL baselines), to prevent non-physical behaviour like atomic fusion during high-energy collisions. The inclusion of an Adaptive Kinetic Monte Carlo (aKMC) module further extends the system's capabilities, allowing it to bridge the gap between the nanosecond timescales of MD and the seconds-to-hours timescales of diffusive phenomena. In essence, PyAceMaker is not just a tool but a comprehensive research assistant that autonomously explores chemical space, learns the underlying physics, and delivers production-ready potentials.

## 2. System Design Objectives

The design of the PyAceMaker system is guided by a set of rigorous objectives, constraints, and success criteria, ensuring that the final product meets the high standards required for scientific research.

### 2.1 Goals

1.  **Democratisation of MLIP Construction**: The system must lower the barrier to entry for MLIP creation. A user should be able to initiate a complex training pipeline with a single configuration file, without needing to write custom Python scripts or understand the intricacies of the underlying machine learning algorithms.
2.  **Autonomous Operation**: The system must be capable of running unattended for extended periods. It should handle errors (such as SCF convergence failures in DFT) gracefully and self-correct without user intervention.
3.  **Physical Robustness**: The generated potentials must be physically safe. They must strictly enforce core repulsion to prevent atoms from overlapping, even in high-energy regimes where training data might be sparse. This is critical for preventing simulation crashes during production runs.
4.  **Data Efficiency**: The system must maximise the information gain from each DFT calculation. By utilising active learning and D-optimality criteria, the system aims to achieve high accuracy with a fraction of the training data required by random sampling methods.
5.  **Multi-Scale Capability**: The architecture must support simulations across different time and length scales, seamlessly integrating MD for fast dynamics and kMC for slow, activated processes.

### 2.2 Constraints

1.  **Computational Resources**: The system must be efficient enough to run on standard cluster nodes. Heavy calculations (DFT) should be offloaded to appropriate queues or external resources if necessary, but the orchestration logic must remain lightweight.
2.  **Dependency Management**: The system relies on external engines like Quantum Espresso, LAMMPS, and Pacemaker. The architecture must abstract these dependencies, allowing for their execution in containers (Docker/Singularity) to ensure reproducibility and ease of deployment.
3.  **Strict Typing and Quality**: The codebase must adhere to strict software engineering standards, utilizing Python 3.11+, strict type hinting (mypy), and rigorous linting (ruff) to ensure maintainability and reduce bugs.
4.  **Security**: As the system executes external commands and handles file I/O, strict validation of paths and inputs is required to prevent security vulnerabilities, especially when running in shared environments.

### 2.3 Success Metrics

1.  **Zero-Config Workflow**: A complete training pipeline (from structure generation to validated potential) must be executable from a single YAML file.
2.  **Accuracy Targets**: The final potentials should achieve a Root Mean Square Error (RMSE) of less than 1 meV/atom for energy and 0.05 eV/Å for forces on the test set.
3.  **Stability**: The system must demonstrate "self-healing" capabilities, successfully recovering from at least 90% of standard DFT convergence errors without human intervention.
4.  **Validation Pass Rate**: Generated potentials must pass physical validation tests (phonon stability, elastic constants, EOS curves) to be considered production-ready.
5.  **Efficiency**: The active learning loop should achieve convergence with significantly fewer DFT evaluations compared to a random sampling baseline.

## 3. System Architecture

The PyAceMaker system follows a modular, microservices-inspired architecture, orchestrated by a central Python application. This design ensures separation of concerns, scalability, and ease of testing.

### 3.1 Components

1.  **The Orchestrator**: The "Brain" of the system. It manages the state of the active learning cycle, coordinates data flow between components, and handles the overall logic of the pipeline (Exploration -> Detection -> Selection -> Refinement).
2.  **Structure Generator**: The "Explorer". Responsible for creating initial atomic configurations and generating local candidates during the refinement phase. It uses an "Adaptive Exploration Policy" to decide whether to use MD, MC, or defect engineering based on the material's properties.
3.  **Oracle**: The "Sage". This component interfaces with First-Principles codes (Quantum Espresso, VASP). It manages the execution of DFT calculations, handling input generation, error correction (e.g., adjusting mixing beta), and output parsing (energy, forces, stress). It also handles "Periodic Embedding" to cut out small clusters from large MD simulations for efficient DFT calculation.
4.  **Trainer**: The "Learner". A wrapper around the Pacemaker engine. It manages the training of the ACE potential, including the selection of the "Active Set" (most informative structures) and the fine-tuning of model weights.
5.  **Dynamics Engine**: The "Executor". Manages the execution of simulations using LAMMPS (for MD) and EON (for kMC). It implements the "On-the-Fly" (OTF) monitoring mechanism, using `fix halt` to stop simulations when the extrapolation grade ($\gamma$) exceeds a safety threshold.
6.  **Validator**: The "Judge". Performs rigorous physical validation of the generated potentials. It calculates phonon dispersion, elastic constants, and equations of state (EOS) to ensure the potential is not just numerically accurate but physically meaningful.

### 3.2 Data Flow

The data flow follows a cyclic "Active Learning" pattern:
1.  **Exploration**: The Dynamics Engine runs a simulation using the current potential.
2.  **Detection**: The simulation is halted if high uncertainty is detected.
3.  **Selection**: The problematic structure is extracted, and local candidates are generated.
4.  **Calculation**: The Oracle performs DFT calculations on these candidates.
5.  **Refinement**: The Trainer updates the potential with the new data.
6.  **Deployment**: The new potential is hot-swapped into the Dynamics Engine, and the simulation resumes.

### 3.3 Diagram

```mermaid
graph TD
    subgraph Control Plane
        Orchestrator[Orchestrator]
        Config[Configuration Manager]
    end

    subgraph Core Modules
        Gen[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Val[Validator]
    end

    subgraph Data Layer
        DB[(Dataset .pckl)]
        PotFiles[Potential Files .yace]
        WorkDir[Working Directory]
    end

    Config --> Orchestrator
    Orchestrator --> Gen
    Orchestrator --> Oracle
    Orchestrator --> Trainer
    Orchestrator --> Dyn
    Orchestrator --> Val

    Gen -- "Candidate Structures" --> Oracle
    Dyn -- "Halted Structures" --> Gen
    Oracle -- "Labelled Data" --> DB
    DB --> Trainer
    Trainer -- "New Potential" --> PotFiles
    PotFiles --> Dyn
    PotFiles --> Val

    classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef control fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef data fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    class Gen,Oracle,Trainer,Dyn,Val core;
    class Orchestrator,Config control;
    class DB,PotFiles,WorkDir data;
```

## 4. Design Architecture

The system is designed with a strong emphasis on type safety and structured data using Pydantic.

### 4.1 File Structure

```
.
├── pyproject.toml
├── README.md
├── src
│   └── mlip_autopipec
│       ├── __init__.py
│       ├── main.py                     # Entry point
│       ├── config.py                   # Pydantic Configuration Models
│       ├── factory.py                  # Component Factory
│       ├── constants.py                # System Constants
│       ├── core
│       │   ├── __init__.py
│       │   ├── orchestrator.py         # Main Logic
│       │   ├── state_manager.py        # State persistence
│       │   └── logger.py               # Centralised logging
│       ├── components
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract Base Classes
│       │   ├── generator
│       │   │   ├── __init__.py
│       │   │   ├── structure_generator.py
│       │   │   └── policies.py         # Adaptive Policies
│       │   ├── oracle
│       │   │   ├── __init__.py
│       │   │   ├── qe_oracle.py        # Quantum Espresso implementation
│       │   │   └── embedding.py        # Periodic Embedding logic
│       │   ├── trainer
│       │   │   ├── __init__.py
│       │   │   └── pacemaker_trainer.py
│       │   ├── dynamics
│       │   │   ├── __init__.py
│       │   │   ├── lammps_driver.py
│       │   │   └── eon_driver.py
│       │   └── validator
│       │       ├── __init__.py
│       │       └── standard_validator.py
│       └── domain_models
│           ├── __init__.py
│           ├── inputs.py               # Structure & Job definitions
│           └── results.py              # Calculation results
└── tests
    ├── unit
    └── integration
```

### 4.2 Data Models

The system relies on a set of robust Domain Models defined in `src/mlip_autopipec/domain_models`. These models serve as the contract between different components, ensuring that data passed between the Oracle, Trainer, and Dynamics engine is well-structured and validated.

*   **`Structure`**: Represents an atomic configuration. It wraps `ase.Atoms` but adds metadata for tracking provenance (e.g., `source: "MD_HALT"`, `cycle: 5`).
*   **`CalculationResult`**: A standardised object returned by the Oracle, containing energy, forces, stress, and convergence status.
*   **`PotentialArtifact`**: Represents a trained potential, including its path, version, and validation metrics.
*   **`ExperimentConfig`**: The root configuration object, parsed from YAML, containing nested configurations for each component (e.g., `OracleConfig`, `TrainerConfig`).

This rigorous data modelling ensures that runtime errors due to malformed data are caught early, and provides clear autocomplete and type checking for developers.

## 5. Implementation Plan

The project is decomposed into 8 sequential implementation cycles. Each cycle builds upon the previous one, adding specific functionality to the system.

### CYCLE 01: Core Framework & Infrastructure
*   **Objective**: Establish the skeleton of the application.
*   **Features**:
    *   Set up project structure, `pyproject.toml`, and CI/CD tools.
    *   Implement the `ConfigurationManager` using Pydantic to parse YAML.
    *   Create the `Orchestrator` shell and logging system.
    *   Define Abstract Base Classes (`BaseGenerator`, `BaseOracle`, etc.) to enforce interfaces.
    *   Implement the `ComponentFactory` for dependency injection.

### CYCLE 02: Structure Generator & Exploration Policy
*   **Objective**: Enable the creation of initial and candidate structures.
*   **Features**:
    *   Implement `StructureGenerator`.
    *   Develop the `AdaptiveExplorationPolicy` engine (determines MD/MC ratios, temperatures).
    *   Implement strategies for random structure generation and defect engineering.
    *   Integrate with ASE for basic crystallographic manipulations.

### CYCLE 03: Oracle (DFT Automation)
*   **Objective**: Implement the interface to First-Principles codes.
*   **Features**:
    *   Implement `QEOracle` (Quantum Espresso wrapper).
    *   Develop the logic for automatic input generation (k-spacing, pseudopotentials).
    *   Implement the self-correction loop for handling SCF convergence errors.
    *   Develop the "Periodic Embedding" logic to cut clusters from MD snapshots for DFT.

### CYCLE 04: Trainer (Pacemaker Integration)
*   **Objective**: Enable the training of ACE potentials.
*   **Features**:
    *   Implement `PacemakerTrainer` wrapper.
    *   Integrate `pace_activeset` for D-optimality based data selection.
    *   Implement the training loop with support for fine-tuning (`--initial_potential`).
    *   Manage dataset files (`.pckl`, `.gzip`).

### CYCLE 05: Dynamics Engine (LAMMPS)
*   **Objective**: Enable basic MD simulations with hybrid potentials.
*   **Features**:
    *   Implement `LAMMPSDriver` using `lammps` Python module or file-based interface.
    *   Implement logic to generate `pair_style hybrid/overlay` commands (ACE + ZBL/LJ).
    *   Ensure robust execution of MD runs with proper cleanup.

### CYCLE 06: On-the-Fly (OTF) Loop
*   **Objective**: Close the Active Learning loop.
*   **Features**:
    *   Implement the "Watchdog" in LAMMPS (`fix halt` based on `v_max_gamma`).
    *   Implement the logic to extract halted structures (`Halt & Diagnose`).
    *   Connect the loop: MD -> Halt -> Extract -> Embed -> DFT -> Train -> Resume.
    *   This is the core "Orchestrator" logic completion.

### CYCLE 07: Advanced Dynamics (EON/kMC)
*   **Objective**: Extend capabilities to long timescales.
*   **Features**:
    *   Implement `EONDriver` for Adaptive Kinetic Monte Carlo.
    *   Create the Python driver script for EON to call the ACE potential.
    *   Implement the bridge between MD deposition and kMC ordering.
    *   Integrate OTF checks within the kMC steps.

### CYCLE 08: Validator & Quality Assurance
*   **Objective**: Implement the final quality gate.
*   **Features**:
    *   Implement `StandardValidator`.
    *   Develop tests for Phonon dispersion (stability check).
    *   Develop tests for Elastic constants and EOS curves.
    *   Generate the HTML validation report.
    *   Final system integration and polish.

## 6. Test Strategy

A comprehensive test strategy is essential for a system of this complexity. Testing will be performed at multiple levels.

### 6.1 Unit Testing
*   **Scope**: Individual classes and functions.
*   **Tool**: `pytest`.
*   **Strategy**:
    *   Every component (Oracle, Trainer, etc.) must have a corresponding test file.
    *   Logic that depends on external binaries (QE, LAMMPS) will be tested using **Mocks**. For example, `MockOracle` will return a pre-defined `CalculationResult` without running QE.
    *   Pydantic models will be tested for validation logic (e.g., ensuring negative temperatures are rejected).

### 6.2 Integration Testing
*   **Scope**: Interaction between components.
*   **Strategy**:
    *   Test the `Orchestrator`'s ability to call components in sequence.
    *   Verify file I/O operations (e.g., Trainer correctly reading data produced by Oracle).
    *   Test the `ComponentFactory` to ensure correct instantiation from config.

### 6.3 System/End-to-End Testing (UAT)
*   **Scope**: Full workflow execution.
*   **Strategy**:
    *   **Mock Mode**: A full system run where all external engines are mocked. This ensures the Python logic (Active Learning loop) works correctly without requiring heavy computation. This will be the primary CI check.
    *   **Real Mode**: Actual execution of the "Fe/Pt on MgO" scenario (as defined in UAT). This verifies the physics and the integration with real external codes.

### 6.4 Regression Testing
*   **Scope**: Ensuring new changes don't break existing functionality.
*   **Strategy**:
    *   Keep a small "Golden Set" of input/output data.
    *   Ensure that refactoring in the `Orchestrator` does not alter the scientific results (within numerical tolerance).

### 6.5 Testing per Cycle
*   **Cycle 01**: Verify config parsing and factory instantiation.
*   **Cycle 02**: Verify structure generation output formats.
*   **Cycle 03**: Verify QE input generation and parser logic (using mock outputs).
*   **Cycle 04**: Verify `pace_train` command generation and dataset manipulation.
*   **Cycle 05**: Verify LAMMPS input script generation.
*   **Cycle 06**: **Critical**: Verify the loop logic. Simulate a "halt" signal and check if the Orchestrator triggers the Oracle.
*   **Cycle 07**: Verify EON config generation.
*   **Cycle 08**: Verify calculation of physical properties against known values (e.g., bulk modulus of Silicon).
