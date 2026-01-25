# System Architecture: PyAcemaker

## 1. Summary

**PyAcemaker** is an advanced, automated system designed to construct and operate Machine Learning Interatomic Potentials (MLIPs), specifically utilizing the **Pacemaker** (Atomic Cluster Expansion - ACE) framework. The primary objective of this project is to democratize the creation of "State-of-the-Art" MLIPs, allowing researchers with limited data science expertise to generate high-fidelity potentials for material science simulations.

In the contemporary landscape of computational materials science, the gap between first-principles accuracy (DFT) and the scalability of classical molecular dynamics (MD) is bridged by MLIPs. However, the traditional workflow for creating these potentials is fraught with challenges: manual iteration, high entry barriers requiring dual expertise in physics and machine learning, and the risk of unphysical extrapolation in unexplored regions. PyAcemaker addresses these issues through a **Zero-Config Workflow**, where a single configuration file orchestrates the entire lifecycle of a potential—from initial structure generation to active learning and final validation.

The system is built upon a robust **Active Learning** cycle. It autonomously explores chemical and structural spaces, detects regions of high uncertainty (extrapolation grade $\gamma$), and selectively performs expensive DFT calculations only on structures that maximize information gain. This "smart" sampling significantly reduces the computational cost compared to random sampling while achieving higher accuracy (Target: Energy RMSE < 1 meV/atom).

A critical feature of PyAcemaker is its focus on **Physics-Informed Robustness**. It implements a hybrid potential strategy, overlaying the Machine Learning potential with a physics-based baseline (Lennard-Jones or ZBL). This ensures that even in regions where the ML model lacks data (e.g., core repulsion during high-energy collisions), the simulation remains physically effectively stable, preventing the "segmentation faults" common in pure MLIP simulations.

Furthermore, the architecture is designed for scalability and extensibility. It supports seamless integration with various simulation engines like **LAMMPS** for molecular dynamics and **EON** for Adaptive Kinetic Monte Carlo (aKMC), enabling the study of phenomena across vast time scales—from picosecond thermal vibrations to second-scale diffusion events. By automating the "Loop" of Exploration, Detection, Calculation, and Refinement, PyAcemaker acts as an autonomous research assistant, continuously improving its understanding of the material system without human intervention.

## 2. System Design Objectives

The design of PyAcemaker is guided by several core objectives and constraints that ensure its effectiveness, reliability, and usability.

### 2.1 Automation and Usability (Zero-Config)
The foremost objective is to minimize human effort. The system must operate as a "black box" capable of taking a chemical composition and a simple configuration file as input and outputting a validated, production-ready potential. This requires:
*   **Self-Healing Workflows:** The system must handle common failures (e.g., DFT convergence errors) autonomously without crashing.
*   **Dynamic Parameter Adjustment:** Hyperparameters for exploration (e.g., temperature schedules) and learning (e.g., regularization) should be determined adaptively based on the material's properties, rather than hardcoded defaults.

### 2.2 Data Efficiency via Active Learning
To achieve high accuracy with minimal computational cost, the system must prioritize "quality over quantity" in data generation.
*   **Uncertainty Quantification:** The system relies on the extrapolation grade ($\gamma$) provided by Pacemaker to identify "unknown" regions.
*   **Active Set Selection:** Utilizing D-Optimality (MaxVol algorithm), the system selects only the most mathematically significant structures for DFT calculation, avoiding redundant computations on correlated data points.
*   **Goal:** Achieve Energy RMSE < 1 meV/atom and Force RMSE < 0.05 eV/Å with 1/10th the DFT cost of random sampling.

### 2.3 Physical Robustness and Safety
A major failure mode of MLIPs is unphysical behavior in extrapolation regimes.
*   **Hybrid Potential Architecture:** The design mandates the use of `pair_style hybrid/overlay` in LAMMPS, combining ACE with ZBL/LJ baselines. This ensures that short-range repulsion is always physical, preventing atomic overlap.
*   **Stability Validation:** The system must rigorously validate generated potentials against physical criteria (Phonon stability, Born stability criteria for elastic constants) before deployment, ensuring that the potential is not just numerically accurate but physically meaningful.

### 2.4 Scalability and Modularity
The architecture must support diverse computational environments and simulation types.
*   **Containerization:** Modules should be designed to run within Docker/Singularity containers, facilitating deployment on HPC clusters or cloud resources.
*   **Engine Agnostic Orchestration:** The core logic (Orchestrator) is decoupled from the specific execution engines (LAMMPS, QE, EON), allowing for future substitution or upgrades of individual components (e.g., switching DFT codes) with minimal refactoring.

## 3. System Architecture

The PyAcemaker system is structured around a central **Orchestrator** that manages the data flow between four specialized modules: the **Structure Generator**, **Oracle**, **Trainer**, and **Dynamics Engine**. This modular design ensures separation of concerns and facilitates independent development and testing of each component.

### 3.1 Component Overview

1.  **Orchestrator (The Brain):**
    *   This Python-based controller manages the workflow state. It decides when to transition between exploration, training, and validation phases.
    *   It handles file management, logging, and error recovery strategies.
    *   It reads the `config.yaml` and instantiates the necessary module drivers.

2.  **Structure Generator (The Explorer):**
    *   Responsible for proposing new atomic configurations.
    *   Implements an **Adaptive Exploration Policy** that selects sampling strategies (MD, MC, Strain, Defects) based on the current uncertainty landscape and material properties.
    *   It prepares inputs for the Dynamics Engine to explore specific regions of the phase space.

3.  **Dynamics Engine (The Executor):**
    *   Runs simulations (MD via LAMMPS, kMC via EON) using the current potential.
    *   **On-the-Fly (OTF) Monitoring:** It continuously monitors the extrapolation grade ($\gamma$). If $\gamma$ exceeds a threshold, it triggers a `fix halt`, stopping the simulation to capture the high-uncertainty structure.
    *   This module is the primary source of "candidate" structures for active learning.

4.  **Oracle (The Sage):**
    *   Performs high-fidelity First-Principles (DFT) calculations.
    *   **Periodic Embedding:** It takes local clusters extracted from the Dynamics Engine and embeds them into periodic supercells suitable for DFT.
    *   **Self-Correction:** It includes logic to automatically retry failed DFT calculations by adjusting mixing parameters or smearing widths.

5.  **Trainer (The Learner):**
    *   Wraps the Pacemaker training tools.
    *   Manages the dataset, merging new DFT data with the existing pool.
    *   Executes `pace_train` to refit the potential.
    *   Performs **Active Set Optimization** to keep the training set compact and informative.

6.  **Validator (The Gatekeeper):**
    *   Runs a suite of physical tests (Phonons, EOS, Elasticity) on the newly trained potential.
    *   Determines if the potential is safe for deployment or requires further refinement.

### 3.2 Data Flow (The Active Learning Cycle)

The data flows in a continuous loop:
1.  **Exploration:** The Orchestrator directs the Dynamics Engine to run a simulation.
2.  **Detection:** The Dynamics Engine detects high uncertainty and halts.
3.  **Selection:** The system extracts the halted structure and generates local perturbations. The Trainer selects the most informative candidates.
4.  **Calculation:** The Oracle computes ground-truth forces and energies for these candidates.
5.  **Refinement:** The Trainer updates the potential using the new data.
6.  **Validation:** The Validator checks the new potential.
7.  **Deployment:** If valid, the new potential is hot-swapped into the Dynamics Engine, and the cycle repeats.

```mermaid
graph TD
    User[User Configuration] -->|Config| Orch(Orchestrator)

    subgraph "Active Learning Cycle"
        Orch -->|Deploy Potential| Dyn[Dynamics Engine<br/>LAMMPS / EON]
        Dyn -->|High Uncertainty Halt| Cands[Candidate Structures]
        Cands -->|Filter (D-Optimality)| Select[Selected Candidates]
        Select -->|Periodic Embedding| Oracle[Oracle<br/>Quantum Espresso]
        Oracle -->|Forces & Energies| Train[Trainer<br/>Pacemaker]
        Train -->|New Potential| Valid[Validator<br/>Phonons/EOS/Elasticity]
        Valid -->|Pass/Fail| Orch
    end

    subgraph "Adaptive Strategy"
        Gen[Structure Generator] -->|Policy: Temp/Pressure/Defects| Dyn
        Orch -->|Feedback| Gen
    end
```

## 4. Design Architecture

The system implementation follows a **Schema-First** approach using **Pydantic**. This ensures strict data validation and clear interfaces between modules. The codebase is organized to reflect the modular architecture.

### 4.1 File Structure

```ascii
src/mlip_autopipec/
├── __init__.py
├── main.py                     # Entry point (CLI)
├── config/                     # Pydantic Configuration Models
│   ├── __init__.py
│   ├── models.py               # Unified config models
│   ├── schemas/                # Detailed schemas per module
│   │   ├── workflow.py
│   │   ├── dft.py
│   │   ├── training.py
│   │   └── validation.py
├── orchestration/              # Workflow Management
│   ├── __init__.py
│   ├── manager.py              # WorkflowManager
│   └── phases/                 # Phase implementations
│       ├── __init__.py
│       ├── exploration.py
│       └── refinement.py
├── dft/                        # Oracle Module
│   ├── __init__.py
│   ├── runner.py               # Abstract DFT Runner
│   ├── qe_runner.py            # Quantum Espresso Implementation
│   └── inputs.py               # Input generation (ASE adapters)
├── trainer/                    # Trainer Module
│   ├── __init__.py
│   ├── pacemaker.py            # Wrapper for pace_train/activeset
│   └── dataset.py              # Data conversion and management
├── dynamics/                   # Dynamics Engine
│   ├── __init__.py
│   ├── lammps.py               # LAMMPS Interface
│   └── eon.py                  # EON Interface (kMC)
├── generator/                  # Structure Generator
│   ├── __init__.py
│   ├── defects.py              # Defect generation logic
│   └── policy.py               # Adaptive Exploration Policy
├── validation/                 # Validation Suite
│   ├── __init__.py
│   ├── runner.py               # Validation Runner
│   ├── phonons.py
│   ├── elasticity.py
│   └── eos.py
└── utils/                      # Utilities
    ├── __init__.py
    ├── logging.py
    └── embedding.py            # Periodic Embedding logic
```

### 4.2 Data Models (Pydantic)

The design relies heavily on strongly typed configuration objects to pass state and parameters.

*   **`WorkflowConfig`**: Top-level configuration defining the project name, working directories, and global limits (e.g., max cycles).
*   **`DFTConfig`**: Defines parameters for the Oracle, including `command` (e.g., `pw.x`), `pseudopotentials` (SSSP paths), `kspacing`, and retry strategies (`mixing_beta` reduction steps).
*   **`TrainingConfig`**: Controls Pacemaker settings: `cutoff` radius, `batch_size`, `max_epochs`, and definitions for the baseline potential (LJ/ZBL).
*   **`InferenceConfig`**: Settings for MD/kMC: `temperature_schedule`, `pressure`, `timestep`, and the critical `uncertainty_threshold` for halting.
*   **`ValidationConfig`**: Thresholds for validation success (e.g., max allowed RMSE, phonon stability requirements).

### 4.3 Key Class Interfaces

*   **`PhaseExecutor`**: An abstract base class for all workflow phases. Each phase (e.g., `DFTPhase`, `TrainingPhase`) implements a `run()` method that takes the current `WorkflowState` and returns an updated state.
*   **`BaseRunner`**: A generic interface for external command execution (LAMMPS, QE), handling `subprocess` calls, timeout management, and error code parsing.
*   **`Validator`**: Interface for validation checks. Each specific check (Phonon, Elasticity) implements `validate(potential_path: Path) -> ValidationResult`.

## 5. Implementation Plan

The project is divided into 8 sequential cycles to ensure a stable and incremental development process.

*   **CYCLE 01: Skeleton & Configuration Infrastructure**
    *   **Goal:** Establish the project foundation.
    *   **Features:**
        *   Project directory structure setup.
        *   Implementation of all Pydantic configuration models (`config/`).
        *   Logging infrastructure setup.
        *   Basic CLI entry point that reads config and initializes the `WorkflowManager`.
        *   Mock implementations of core interfaces.

*   **CYCLE 02: Oracle Phase (DFT Engine)**
    *   **Goal:** Enable automated DFT calculations.
    *   **Features:**
        *   Implementation of `QERunner` for Quantum Espresso.
        *   Input file generation using ASE (Atomic Simulation Environment).
        *   Robust error handling (Self-Correction) for SCF convergence failures.
        *   Parsing of XML/Text outputs to extract Energy, Forces, and Stress.

*   **CYCLE 03: Trainer Phase (Pacemaker Integration)**
    *   **Goal:** Enable ML potential training.
    *   **Features:**
        *   `PacemakerWrapper` to interface with `pace_train`.
        *   Data conversion utilities (ASE atoms to Pacemaker `.pckl.gzip`).
        *   Implementation of `pace_activeset` logic for data selection.
        *   Configuration of LJ/ZBL baselines within the training config.

*   **CYCLE 04: Dynamics Phase I (Basic MD)**
    *   **Goal:** Run Molecular Dynamics with Hybrid Potentials.
    *   **Features:**
        *   `LammpsRunner` implementation.
        *   Generation of `in.lammps` files supporting `pair_style hybrid/overlay`.
        *   Basic NVT/NPT simulation execution.
        *   Parsing of LAMMPS log files and dumps.

*   **CYCLE 05: Dynamics Phase II (Active Learning Loop)**
    *   **Goal:** Implement the "Halt & Diagnose" mechanism.
    *   **Features:**
        *   Integration of `fix halt` based on `compute pace`.
        *   Logic to detect non-zero exit codes from LAMMPS.
        *   `CandidateProcessor` to extract halted structures.
        *   `PeriodicEmbedding` logic to convert clusters to supercells.

*   **CYCLE 06: Validation Suite**
    *   **Goal:** Ensure physical reliability of potentials.
    *   **Features:**
        *   `PhononValidator` using `phonopy`.
        *   `ElasticityValidator` using deformation methods.
        *   `EOSValidator` (Birch-Murnaghan fit).
        *   Generation of `validation_report.html`.

*   **CYCLE 07: Advanced Exploration (Structure Generator)**
    *   **Goal:** Smarter sampling beyond random MD.
    *   **Features:**
        *   `StructureGenerator` module.
        *   Implementation of `DefectStrategy` (Vacancies, Interstitials).
        *   `AdaptivePolicy` logic to adjust MD parameters based on feedback.
        *   High-temperature / High-pressure sampling routines.

*   **CYCLE 08: Integration & EON (Full System)**
    *   **Goal:** Complete system with kMC and full orchestration.
    *   **Features:**
        *   `EONWrapper` for Adaptive Kinetic Monte Carlo.
        *   `pace_driver.py` for external potential interface in EON.
        *   Final integration of all phases into the `WorkflowManager` loop.
        *   Comprehensive end-to-end testing and documentation.

## 6. Test Strategy

Testing is integral to the AC-CDD methodology. Each cycle will include specific test deliverables.

### 6.1 Unit Testing Strategy
*   **Framework:** `pytest` will be used for all unit tests.
*   **Coverage:** Target 100% coverage for logic-heavy modules (e.g., `utils`, `config`, `orchestration`).
*   **Mocks:** Heavy use of `unittest.mock` to simulate external binaries (LAMMPS, QE, Pacemaker). We will not run actual heavy physics simulations in unit tests.
    *   Mocking `subprocess.run` to return fake stdout/stderr for parsing logic.
    *   Mocking file I/O for configuration reading.

### 6.2 Integration Testing Strategy
*   **Component Tests:** For each runner (QE, LAMMPS), we will have integration tests that run small, fast examples if the binaries are available in the environment (skipped otherwise).
*   **Data Flow Tests:** Verifying that data objects (ASE Atoms) are correctly transformed between modules (e.g., Atoms -> Lammps Data -> Atoms -> DFT Input).

### 6.3 Cycle-Specific Testing
*   **Cycle 01:** Test configuration loading and validation logic.
*   **Cycle 02:** Test DFT input generation correctness (text comparison) and parser accuracy (using sample output files).
*   **Cycle 03:** Test dataset file creation and Pacemaker command construction.
*   **Cycle 04:** Test LAMMPS input file generation (especially hybrid pair styles).
*   **Cycle 05:** Test the "Halt" detection logic by feeding a log file with a halt event. Test Periodic Embedding on dummy structures.
*   **Cycle 06:** Test validation logic using pre-calculated dummy potentials (or mock results).
*   **Cycle 07:** Test defect generation symmetry (using `spglib` if needed) and policy decision trees.
*   **Cycle 08:** End-to-End "Dry Run" test. Execute the full orchestrator loop with all external calls mocked to verify the sequence of operations.

### 6.4 User Acceptance Testing (UAT)
*   **Scenario-Based:** Each cycle includes a UAT plan defined in `UAT.md`.
*   **Jupyter Notebooks:** We will provide `.ipynb` files for users to interactively run the newly implemented features (e.g., visualize a generated structure, inspect a validation report).
*   **Behavior Driven:** Scenarios will be described in Gherkin (Given/When/Then) format to ensure alignment with business requirements.
