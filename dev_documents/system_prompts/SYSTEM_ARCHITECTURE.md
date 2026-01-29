# System Architecture for PyAcemaker

## 1. Summary

PyAcemaker is a cutting-edge, automated software system designed to construct Machine Learning Interatomic Potentials (MLIP) with high efficiency and reliability. The core purpose of this system is to democratise the creation of "State-of-the-Art" potentials, specifically using the Atomic Cluster Expansion (ACE) framework via the Pacemaker engine. In the field of computational materials science, bridging the gap between the high accuracy of Density Functional Theory (DFT) and the large-scale capabilities of Molecular Dynamics (MD) has always been a significant challenge. MLIPs offer a solution, but their construction typically requires deep expertise in both data science and physics, involving tedious manual cycles of structure generation, calculation, training, and verification. PyAcemaker removes these barriers by providing a "Zero-Config" workflow that automates the entire pipeline.

The system is built to address three critical issues in the traditional MLIP workflow. Firstly, it tackles the problem of biased sampling. Standard MD simulations often fail to capture rare events or high-energy configurations necessary for a robust potential. PyAcemaker employs an active learning strategy that specifically targets these high-uncertainty regions, ensuring the potential is valid even in extreme conditions. Secondly, it solves the issue of data inefficiency. Instead of calculating thousands of redundant structures with expensive DFT, the system intelligently selects only the most informative atomic configurations. This approach can reduce the computational cost by an order of magnitude while maintaining or exceeding the accuracy of random sampling. Thirdly, it drastically lowers the maintenance cost of potentials. By implementing self-healing loops, the system can automatically detect when a simulation is about to fail due to unphysical predictions, pause the simulation, learn from the problem, and resume with an improved model.

PyAcemaker operates as a comprehensive ecosystem. It includes an "Orchestrator" that manages the workflow, a "Structure Generator" that acts as an explorer finding new atomic arrangements, an "Oracle" that performs the ground-truth quantum mechanical calculations, a "Trainer" that fits the mathematical models, and a "Dynamics Engine" that runs the actual simulations. This modular design ensures that each part of the system can be improved or replaced without breaking the whole. The system is designed to be deployed easily, using containerisation technologies like Docker, making it accessible for researchers on local workstations as well as on High-Performance Computing (HPC) clusters.

Ultimately, PyAcemaker aims to transform the way materials are discovered and simulated. By automating the complex task of potential generation, it frees researchers to focus on the science of materials—predicting properties, discovering new alloys, and understanding reaction mechanisms—rather than wrestling with the tools of the trade. It is a robust, scalable, and user-friendly platform that brings the power of machine learning to the fingertips of every materials scientist.

## 2. System Design Objectives

The design of PyAcemaker is guided by several ambitious objectives and strict constraints to ensure it meets the needs of modern materials research. These objectives serve as the success criteria for the project and define the boundaries within which the system must operate.

**Zero-Config Workflow (Automation)**
The primary objective is to achieve a "Zero-Config" user experience. A user should be able to start a complex active learning campaign by providing a single configuration file (YAML) describing the material of interest (e.g., "Ti-O system"). The system must handle all subsequent steps without human intervention. This includes generating initial structures, setting up DFT calculations, handling calculation errors (such as SCF convergence failures), training the potential, and iterating the loop. The goal is to reduce the human time spent on these tasks to near zero, allowing the software to run 24/7 autonomously.

**Data Efficiency (Active Learning)**
A key constraint in materials science is the high computational cost of DFT calculations. PyAcemaker aims to maximise "Data Efficiency". The target is to achieve a high-fidelity potential (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) using less than 10% of the data required by traditional random sampling methods. This is achieved through active learning, where the system calculates the "uncertainty" of the potential for every atom in a simulation. Only structures that exhibit high uncertainty—indicating that the physics is not well-understood in that region—are selected for DFT calculation. This ensures that every expensive CPU/GPU hour spent on DFT contributes directly to improving the potential.

**Physics-Informed Robustness (Safety)**
Machine learning models are mathematical approximators and can behave unpredictably outside their training data (extrapolation). In atomistic simulations, this can be catastrophic, leading to atoms overlapping (nuclear fusion) or flying apart unphysically. PyAcemaker enforces "Physics-Informed Robustness" by incorporating physical baselines. It uses a "Delta Learning" approach where a standard physics-based potential (like Lennard-Jones or ZBL) handles the short-range repulsion and basic bonding, while the ML model learns the complex many-body corrections. This ensures that even in unknown regions, the system behaves safely and does not crash due to unphysical forces. The system must guarantee that simulations can recover from high-uncertainty events without segmentation faults.

**Scalability and Extensibility**
The architecture must be scalable from a single workstation to massive HPC environments. The "Dynamics Engine" must support not just small-scale MD but also large-scale simulations with millions of atoms. Furthermore, the system must be extensible to different time scales. It should integrate not only with Molecular Dynamics (MD) for nanosecond phenomena but also with Adaptive Kinetic Monte Carlo (aKMC) for investigating diffusion and reactions over seconds or hours. The modular design should allow future integration of other DFT codes (like VASP or CASTEP) or other ML potentials beyond ACE, although ACE is the current focus.

## 3. System Architecture

The architecture of PyAcemaker is designed around a central "Orchestrator" that coordinates the activities of specialized modules. These modules interact through well-defined interfaces, exchanging data and control signals to execute the Active Learning Cycle.

### High-Level Components

1.  **Orchestrator**: The "Brain" of the system. It reads the user configuration, manages the state of the workflow (Exploration, Selection, Calculation, Refinement), and directs the other modules. It ensures data flows correctly and handles the overall logic of the loop.

2.  **Structure Generator**: The "Explorer". Its role is to propose new atomic structures. Instead of random guessing, it uses an "Adaptive Exploration Policy". Depending on the material (e.g., metal vs insulator) and the current state of knowledge, it decides whether to run high-temperature MD, introduce defects (vacancies, interstitials), or apply strain. It generates candidates that are likely to expose weaknesses in the current potential.

3.  **Dynamics Engine**: The "Runner". This module runs the actual simulations using LAMMPS (for MD) or EON (for kMC). It is equipped with an "Uncertainty Monitor". As the simulation runs, it calculates the extrapolation grade ($\gamma$) for every atom. If $\gamma$ exceeds a safety threshold, it halts the simulation and flags the structure as "unsafe" and "interesting" for learning.

4.  **Oracle**: The "Sage". This module provides the ground truth. It manages the Density Functional Theory (DFT) calculations using Quantum Espresso. It includes a "Self-Healing" mechanism to automatically fix common calculation errors (like mixing beta adjustments) to ensure that valid training data is returned without user intervention. It also handles "Periodic Embedding", creating small, computationally cheap simulation boxes around the interesting local structures found by the Dynamics Engine.

5.  **Trainer**: The "Learner". This module interfaces with the Pacemaker engine. It manages the dataset, selects the most informative structures using "Active Set Optimization" (D-Optimality), and fits the ACE potential. It ensures the potential fits the difference between the DFT data and the physical baseline (Delta Learning).

6.  **Validator**: The "Judge". Before a new potential is deployed, this module subjects it to rigorous testing. It checks not just numerical errors (RMSE) but physical properties like phonon stability, elastic constants, and Equation of State (EOS) curves. Only potentials that pass these physics checks are promoted to the next generation.

### System Interaction Diagram

```mermaid
graph TD
    User[User Config (YAML)] --> Orch{Orchestrator}
    Orch -->|1. Explore| Gen[Structure Generator]
    Orch -->|1. Explore| Dyn[Dynamics Engine]

    Dyn -->|Halt on High Uncertainty| Orch
    Gen -->|Candidate Structures| Orch

    Orch -->|2. Select| Trainer[Trainer / Active Set]
    Trainer -->|Selected Candidates| Oracle[Oracle (DFT)]

    Oracle -->|3. Compute (Energy/Forces)| DB[(Database)]
    DB --> Trainer

    Trainer -->|4. Train| Pot[Potential (YACE)]
    Pot -->|5. Validate| Val[Validator]

    Val -- Pass --> Orch
    Val -- Fail --> Gen

    Dyn -.->|Uses| Pot
```

### Data Flow (The Active Learning Cycle)

The system operates in a continuous loop:
1.  **Exploration**: The Dynamics Engine runs MD/kMC using the current potential. It monitors uncertainty.
2.  **Detection**: If uncertainty is high, the simulation halts. The problematic structure is extracted.
3.  **Selection**: The Trainer analyzes the structure and selects the most informative local environments (using D-Optimality) to add to the training set.
4.  **Calculation**: The Oracle runs DFT on these selected small structures (embedded in periodic boxes).
5.  **Refinement**: The Trainer updates the potential using the new data.
6.  **Validation**: The Validator checks the new potential. If good, it replaces the old one, and the Dynamics Engine resumes the simulation.

## 4. Design Architecture

The system uses a clean, modular file structure and rigorous data modelling using Pydantic to ensure type safety and configuration validity.

### File Structure

```ascii
mlip_autopipec/
├── config/
│   ├── schemas/
│   │   ├── common.py      # Common types (Element, Path)
│   │   ├── dft.py         # DFT settings (QE, VASP)
│   │   ├── workflow.py    # Orchestrator settings
│   │   └── validation.py  # Validation thresholds
│   └── loader.py          # YAML loading logic
├── orchestration/
│   ├── loop.py            # Main active learning loop
│   └── state.py           # State management (checkpoints)
├── generator/
│   ├── policy.py          # Adaptive Exploration Policy
│   ├── defects.py         # Vacancy/Interstitial generation
│   └── distortions.py     # Strain/EOS generation
├── dft/
│   ├── runner.py          # QE execution wrapper
│   ├── error_handler.py   # Self-correction logic
│   └── embedding.py       # Periodic embedding logic
├── trainer/
│   ├── pacemaker.py       # Wrapper for pace_train, pace_activeset
│   └── dataset.py         # Data management (pickles)
├── dynamics/
│   ├── lammps.py          # LAMMPS interface (watchdog)
│   ├── eon.py             # EON (kMC) interface
│   └── uncertainty.py     # Gamma monitoring logic
├── validation/
│   ├── phonons.py         # Phonopy integration
│   ├── elastic.py         # Elastic constants calculation
│   └── eos.py             # Equation of State checks
└── app.py                 # CLI Entry point
```

### Data Models and Domain Concepts

The design relies heavily on "Configuration as Code". The `config` module defines the entire system state.

-   **`WorkflowConfig`**: Defines the overall parameters of the active learning campaign (e.g., maximum cycles, uncertainty thresholds, target RMSE).
-   **`DFTConfig`**: Encapsulates all details needed for the Oracle. It ensures that pseudopotentials are defined, K-point spacing is valid, and command paths are secure.
-   **`Structure`**: A core domain object wrapping `ase.Atoms`. It carries metadata such as "tags" (e.g., 'liquid', 'high_gamma', 'transition_state') and provenance information (which cycle generated it).
-   **`Potential`**: Represents a versioned MLIP artifact. It tracks its parent dataset, the training metrics, and its validation status.

**Key Invariants**:
-   **Safe Execution**: All external commands (LAMMPS, QE, Pacemaker) are executed with `shell=False` to prevent injection vulnerabilities.
-   **Atomic Consistency**: Data added to the training set is strictly validated for NaN/Inf values and physical sanity (minimum interatomic distance).
-   **Reproducibility**: Every training run saves a full snapshot of the configuration and random seeds, ensuring results can be reproduced.

## 5. Implementation Plan

The development is divided into 6 strictly defined cycles.

### Cycle 01: Core Framework & Oracle (DFT)
**Goal**: Establish the project skeleton and a working DFT engine.
**Features**:
-   Setup project structure, `pyproject.toml`, and dependency management.
-   Implement the `config` module with Pydantic schemas.
-   Develop the `Oracle` module: The `QERunner` class capable of running Quantum Espresso.
-   Implement `ErrorHandler` for DFT to handle basic SCF convergence failures (adjusting mixing beta).
-   Create a basic `Orchestrator` shell that can read config and trigger a simple DFT calculation.
-   **Deliverable**: A system that can take a structure and reliably return its DFT energy and forces.

### Cycle 02: Structure Generation & Database Management
**Goal**: Enable the system to generate training candidates and manage data.
**Features**:
-   Implement the `StructureGenerator` module. Focus on "Initial Exploration" using M3GNet or random perturbations to create starting structures.
-   Develop `DatabaseManager` to handle `ase.Atoms` objects, saving them to disk (Pickle/HDF5) and managing unique IDs.
-   Implement the `Trainer` module's basic wrapper for Pacemaker (`pace_train`).
-   Enable the loop to: Generate Structure -> Run DFT -> Save to DB -> Train initial potential.
-   **Deliverable**: A functioning "One-Shot" training pipeline.

### Cycle 03: Dynamics Engine (LAMMPS MD) & Uncertainty
**Goal**: Implement the inference engine with safety mechanisms.
**Features**:
-   Develop `LammpsRunner` in the `dynamics` module.
-   Implement "Hybrid Potential" logic: Automatically generating LAMMPS input scripts that overlay ACE with ZBL/LJ baselines for safety.
-   Implement "Uncertainty Monitoring": Use `fix halt` in LAMMPS to stop simulation when gamma exceeds threshold.
-   Implement logic to parse LAMMPS logs and extract the specific "Halted Structure".
-   **Deliverable**: A LAMMPS run that self-terminates when it encounters unknown physics.

### Cycle 04: Active Learning Loop (Orchestrator Integration)
**Goal**: Close the loop. Automate the cycle of Explore-Detect-Select-Refine.
**Features**:
-   Integrate Cycle 03's halting mechanism into the Orchestrator.
-   Implement `CandidateProcessor`: When MD halts, extract the structure, create "Local Candidates" (small perturbations), and apply "Periodic Embedding" to make them suitable for DFT.
-   Implement "Local D-Optimality": Use `pace_activeset` to select only the most informative of these candidates.
-   Automate the "Resume" logic: After re-training, restart the MD from the halt point with the new potential.
-   **Deliverable**: A fully autonomous Active Learning loop that improves the potential on-the-fly.

### Cycle 05: Validation Framework
**Goal**: Ensure quality control.
**Features**:
-   Develop the `Validator` module.
-   Implement `PhononCheck`: Integrate with Phonopy to check for imaginary frequencies.
-   Implement `ElasticCheck`: Calculate elastic constants and check Born stability criteria.
-   Implement `EOSCheck`: Fit Birch-Murnaghan curves.
-   Generate HTML reports summarizing the validation results (Pass/Fail/Conditional).
-   **Deliverable**: A gatekeeper system that prevents bad potentials from being accepted.

### Cycle 06: Advanced Dynamics (kMC), Adaptive Policy & Final Integration
**Goal**: Expand capabilities and intelligence.
**Features**:
-   Implement `EONWrapper` to support Adaptive Kinetic Monte Carlo (aKMC) for long-time-scale exploration.
-   Develop the "Adaptive Exploration Policy Engine": A logic unit that decides whether to run MD, kMC, or High-T sampling based on the material type and current uncertainty stats.
-   Finalise the CLI (`mlip-auto`) and end-to-end integration.
-   Comprehensive documentation and tutorials.
-   **Deliverable**: The complete PyAcemaker system v1.0.

## 6. Test Strategy

Testing is a critical part of the AC-CDD methodology. Each cycle has a dedicated test strategy.

### Cycle 01 Testing
-   **Unit Tests**: Verify Pydantic schemas reject invalid configs. Test `QERunner` input generation (checking text output against golden files).
-   **Mocking**: Heavily mock the actual `pw.x` executable to test error handling logic (e.g., simulate a crash and check if `ErrorHandler` retries with new params).
-   **Integration**: Run a real, tiny DFT calculation (e.g., a single Silicon atom) if the environment allows, to verify the binary interface.

### Cycle 02 Testing
-   **Unit Tests**: Verify `StructureGenerator` produces valid ASE objects. Check `DatabaseManager` for data consistency (no NaNs).
-   **Integration**: Run the full "One-Shot" pipeline: Generate -> (Mock DFT) -> Train. Verify that `pace_train` is called with correct arguments and produces a dummy potential file.
-   **Data Integrity**: Check that metadata (tags, provenance) is correctly preserved across the pipeline.

### Cycle 03 Testing
-   **Unit Tests**: Verify `LammpsInputWriter` correctly generates `hybrid/overlay` commands.
-   **Simulation**: Create a synthetic test case where a potential has a known "hole" (high uncertainty). Run LAMMPS and assert that it halts exactly at that point.
-   **Safety**: Verify that running with ZBL baseline prevents atoms from collapsing (distance -> 0) even with random forces.

### Cycle 04 Testing
-   **End-to-End**: Run a "Mini-Loop". Start with a blank potential, run MD, trigger a mock halt, select structures, run mock DFT, retrain.
-   **State Recovery**: Interrupt the orchestrator process in the middle of a cycle and restart it. Verify it resumes from the correct state (e.g., "Waiting for DFT").
-   **Performance**: measure the overhead of the "Selection" step. It must be fast enough not to bottleneck the loop.

### Cycle 05 Testing
-   **Scientific Validation**: Use known good potentials (e.g., standard Si potential) and bad potentials (random weights). Assert that the `Validator` correctly passes the good one and fails the bad one.
-   **Phonon Tests**: Verify that `PhononCheck` correctly identifies imaginary modes in unstable structures.
-   **Reporting**: Check that the generated HTML report renders correctly and contains all necessary plots.

### Cycle 06 Testing
-   **System Test**: Run a full production-like scenario (e.g., Al-Cu alloy) for 24 hours.
-   **Policy Check**: Verify that the "Adaptive Policy" changes strategies (e.g., switches from MD to kMC) when appropriate signals are detected.
-   **User Acceptance (UAT)**: Walk through the "Zero-Config" user stories. Ensure a new user can install and run the system with only the documentation provided.
