# System Architecture Design for PYACEMAKER

## 1. Summary

The PYACEMAKER project represents a paradigm shift in the construction and operation of Machine Learning Interatomic Potentials (MLIPs). In the contemporary landscape of computational materials science, the dichotomy between the high accuracy of First-Principles calculations (such as Density Functional Theory - DFT) and the scalability of Classical Molecular Dynamics (MD) has long been a bottleneck. MLIPs bridge this gap by learning the quantum mechanical potential energy surface (PES) and reproducing it at a fraction of the computational cost. However, the creation of a robust, production-grade MLIP typically requires deep expertise in both data science and computational physics, involving a tedious, manual workflow of structure sampling, DFT calculation, model training, and validation.

PYACEMAKER aims to democratize this technology by providing a "Zero-Config" automation system. It is designed to allow researchers, even those with limited ML expertise, to generate "State-of-the-Art" MLIPs using the Atomic Cluster Expansion (ACE) formalism (via the Pacemaker engine). The system is not merely a wrapper around existing tools but an intelligent Orchestrator that manages the entire lifecycle of a potential. It autonomously explores chemical and structural spaces, detects regions of high uncertainty (extrapolation), and selectively refines the potential using Active Learning strategies.

The core philosophy of PYACEMAKER addresses three critical challenges in the field:
1.  **Sampling Bias and Extrapolation Risk**: Standard equilibrium MD simulations often fail to sample rare events or high-energy configurations (e.g., transition states, defects). When a learned potential encounters these "unknown" regions during operation, it can behave unphysically. PYACEMAKER employs an "Adaptive Exploration Policy" that dynamically adjusts sampling strategies (MD vs. Monte Carlo, temperature schedules, defect injection) to proactively cover these blind spots.
2.  **Data Efficiency**: Brute-force generation of DFT data is computationally prohibitive. PYACEMAKER utilizes D-Optimality (MaxVol algorithm) to select only the most information-rich structures for labeling, achieving high accuracy with a fraction of the training data compared to random sampling.
3.  **Operational Robustness**: The system integrates physics-based safeguards. By employing a hybrid potential scheme (ACE + Physics Baseline like Lennard-Jones or ZBL), it ensures that the simulation never catastrophically fails due to non-physical attractive forces in the core-repulsion region, a common issue in polynomial-based potentials.

Furthermore, PYACEMAKER is built for scalability. It seamlessly transitions from short-term MD simulations to long-term evolution using Adaptive Kinetic Monte Carlo (aKMC) via EON integration. This allows the same potential to be trained on and applied to phenomena ranging from picosecond lattice vibrations to second-scale diffusion and ordering processes. The architecture supports deployment across diverse environments, from local workstations to High-Performance Computing (HPC) clusters, by encapsulating complexity within modular, container-friendly components.

In essence, PYACEMAKER transforms the "art" of potential fitting into a rigorous, automated engineering process, enabling materials scientists to focus on physical discovery rather than parameter tuning.

## 2. System Design Objectives

The architectural decisions for PYACEMAKER are guided by four primary objectives, each defining specific constraints and success criteria for the system.

### 2.1. Zero-Config Workflow (Automation & Usability)
**Goal**: The system must operate autonomously from an initial configuration to a final, validated potential.
**Constraint**: The user interface should be restricted to a single declarative configuration file (`config.yaml`) and an initial structure file. No custom Python scripting should be required for standard workflows.
**Success Criteria**: A user can define "Fe-Pt alloy" and a target accuracy (e.g., RMSE Energy < 1 meV/atom), and the system autonomously performs initial exploration, active learning loops, and validation without human intervention.
**Strategy**: Implement a central `Orchestrator` that manages state transitions. All component behaviors (Generator, Oracle, Trainer) must be configurable via Pydantic models derived from the YAML input.

### 2.2. Data Efficiency via Active Learning
**Goal**: Minimize the number of expensive DFT calculations required to reach target accuracy.
**Constraint**: The system must not blindly accumulate data. It must strictly filter redundant structures.
**Success Criteria**: Achieve the same validation accuracy as a random sampling approach using less than 10% of the DFT computational cost.
**Strategy**: Implement "Uncertainty Quantification" (UQ) using the extrapolation grade ($\gamma$) provided by Pacemaker. Couple this with "Active Set Selection" (D-Optimality) to select a minimal set of basis structures that maximize the information determinant of the design matrix.

### 2.3. Physics-Informed Robustness
**Goal**: Prevent "garbage out" scenarios where the potential predicts non-physical behavior in unlearned regions.
**Constraint**: The potential must always exhibit positive divergence in energy as interatomic distances approach zero (core repulsion).
**Success Criteria**: Zero "Segmentation Faults" or simulation explosions during high-temperature MD or radiation damage simulations.
**Strategy**: Enforce a "Delta Learning" architecture. The Machine Learning model (ACE) learns the *difference* between the DFT energy and a physics-based baseline (LJ/ZBL). In the final deployment, the system generates LAMMPS `hybrid/overlay` commands to explicitly sum these terms, ensuring the physics baseline dominates at short range.

### 2.4. Scalability & Extensibility
**Goal**: Support multi-scale simulations (Time and Length) and diverse computing environments.
**Constraint**: The system must handle the "Time-Scale Problem" by bridging MD and kMC. It must also abstract the underlying execution environment (Local vs Slurm).
**Success Criteria**: A single potential trained by the system can be used for both MD (LAMMPS) and kMC (EON) without manual format conversion.
**Strategy**: Use modular interfaces (`Driver` pattern) for external codes. The `DynamicsEngine` treats LAMMPS and EON as interchangeable "Explorers" that feed data back into the active learning loop.

## 3. System Architecture

The system follows a modular, cycle-based architecture centered around an **Orchestrator**.

### High-Level Component Diagram

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]

    subgraph "Core Components"
        Orch --> Gen[Structure Generator]
        Orch --> Dyn[Dynamics Engine]
        Orch --> Ora[Oracle (DFT)]
        Orch --> Trn[Trainer (Pacemaker)]
        Orch --> Val[Validator]
    end

    subgraph "External Tools"
        Gen -.-> Pym[Pymatgen / ASE]
        Dyn -.-> LAMMPS[LAMMPS (MD)]
        Dyn -.-> EON[EON (aKMC)]
        Ora -.-> QE[Quantum Espresso / VASP]
        Trn -.-> PACE[Pacemaker Core]
        Val -.-> Phon[Phonopy]
    end

    subgraph "Data Flow"
        Gen -- "Candidates" --> Ora
        Dyn -- "Halted Structures" --> Gen
        Ora -- "Labeled Data" --> Trn
        Trn -- "Potential.yace" --> Dyn
        Trn -- "Potential.yace" --> Val
    end
```

### Component Details

1.  **Orchestrator**: The brain of the system. It runs the main loop (Exploration -> Detection -> Selection -> Refinement). It manages the file system state (`active_learning/iter_XXX`), handles error recovery, and decides when to transition between phases.
2.  **Structure Generator**: The "Explorer". It proposes new atomic configurations. It implements an `AdaptiveExplorationPolicy` that chooses between Random Search, High-T MD, MC (Swap), or Defect injection based on the current knowledge state.
3.  **Dynamics Engine**: The "Verifier" and "User". It runs simulations using the current potential. It features an "On-the-Fly" (OTF) watchdog that monitors the extrapolation grade ($\gamma$). If $\gamma$ exceeds a threshold, it halts the simulation (`fix halt`) and returns the problematic structure to the Orchestrator. It supports both LAMMPS (for MD) and EON (for kMC).
4.  **Oracle**: The "Teacher". It performs DFT calculations. It includes a "Self-Correction" mechanism to handle SCF convergence failures automatically. It also handles "Periodic Embedding", converting local clusters cut from MD into periodic supercells suitable for DFT.
5.  **Trainer**: The "Learner". It wraps the Pacemaker engine. It manages the training dataset, performs Active Set Selection (MaxVol) to filter data, and executes the fitting process using Delta Learning (learning residuals from a Physics Baseline).
6.  **Validator**: The "Judge". It performs rigorous physical validation (Phonons, Elastic Constants, EOS, Melting Point) to ensure the potential is not just numerically accurate but physically meaningful.

## 4. Design Architecture

The system is designed using strict Python typing and Pydantic models to ensure robustness and clarity.

### File Structure (Proposed)

```
src/mlip_autopipec/
├── __init__.py
├── main.py                     # Entry point
├── core/
│   ├── orchestrator.py         # Main loop logic
│   ├── state_manager.py        # Resume/Checkpoint handling
│   └── exceptions.py
├── domain_models/              # Pydantic Schemas
│   ├── config.py               # YAML configuration model
│   ├── structure.py            # Atom/Structure abstraction
│   ├── potential.py            # Potential metadata
│   └── enums.py
├── components/
│   ├── generator/
│   │   ├── base.py
│   │   ├── policy.py           # Adaptive logic
│   │   └── builder.py          # Random/Defect generation
│   ├── oracle/
│   │   ├── base.py
│   │   ├── qe_driver.py        # Quantum Espresso Interface
│   │   ├── vasp_driver.py      # VASP Interface
│   │   └── embedder.py         # Periodic Embedding Logic
│   ├── trainer/
│   │   ├── base.py
│   │   ├── pacemaker_driver.py
│   │   └── active_set.py       # D-Optimality Logic
│   ├── dynamics/
│   │   ├── base.py
│   │   ├── lammps_driver.py    # MD Interface
│   │   ├── eon_driver.py       # kMC Interface
│   │   └── hybrid.py           # Hybrid Potential Utils
│   └── validator/
│       ├── base.py
│       ├── phonon.py
│       └── elastic.py
└── interfaces/                 # Adapters for External Libraries
    ├── ase_adapter.py
    └── lammps_adapter.py
```

### Key Data Models

*   **`Config`**: The root configuration object, validated against the user's YAML. It contains sub-configs for `GeneratorConfig`, `OracleConfig`, etc.
*   **`Structure`**: An abstraction over `ase.Atoms`, carrying additional metadata like `provenance` (how it was generated), `tags` (defect info), and `labels` (DFT energy/forces).
*   **`Potential`**: Represents a trained model version. It stores the path to the `.yace` file, the `PhysicsBaseline` used (e.g., LJ parameters), and validation metrics.
*   **`CycleReport`**: A summary object generated at the end of each active learning cycle, containing statistics on added structures, training errors, and validation outcomes.

## 5. Implementation Plan

The project will be executed in 8 strict sequential cycles.

### CYCLE 01: Core Framework & Configuration
*   **Goal**: Establish the project skeleton, configuration management, and the base Orchestrator.
*   **Features**:
    *   Define Pydantic models for `Config`, `Structure`, and `Potential`.
    *   Implement `StateManager` to handle directory creation and resumption of jobs.
    *   Create the abstract base classes for all major components (`BaseGenerator`, `BaseOracle`, `BaseTrainer`, `BaseDynamics`).
    *   Implement a mock Orchestrator loop that validates the configuration and initializes the file system.
*   **Outcome**: A runnable `main.py` that reads a `config.yaml`, validates it, and sets up the folder structure (`active_learning/`, `potentials/`).

### CYCLE 02: Structure Generator & Adaptive Policy
*   **Goal**: Implement the logic to create initial structures and explore chemical space.
*   **Features**:
    *   Implement `StructureGenerator` with strategies for: Random Perturbation, Supercell generation, and Defect insertion (Vacancy/Interstitial).
    *   Develop the `AdaptiveExplorationPolicy` engine that interprets the current cycle state to decide *how* to generate structures (e.g., "If initial cycle, use Random; if later, use Defect").
    *   Implement "Cold Start" capability using M3GNet (optional) or simple randomization to seed the first generation.
*   **Outcome**: The system can generate a diverse set of `ase.Atoms` objects ready for DFT calculation.

### CYCLE 03: Oracle (DFT Automation)
*   **Goal**: robustly generate reference data using DFT codes.
*   **Features**:
    *   Implement `QEOracle` (Quantum Espresso) and `VASPOracle`.
    *   Develop the "Self-Correction" logic: a loop that catches DFT errors (SCF convergence) and retries with adjusted parameters (mixing beta, smearing).
    *   Implement `PeriodicEmbedder`: logic to take a non-periodic cluster (from MD halt) and embed it into a periodic box with vacuum or buffer for correct DFT evaluation.
*   **Outcome**: The system can take a list of `Structure` objects, run DFT, and return them with calculated Forces and Energies, handling errors gracefully.

### CYCLE 04: Trainer (Pacemaker Integration)
*   **Goal**: Automate the fitting of the ACE potential.
*   **Features**:
    *   Implement `PacemakerTrainer` to wrap the `pace_train` and `pace_activeset` executables.
    *   Implement "Delta Learning" configuration: automatically generating the YAML config for Pacemaker to learn only the residual against a reference (LJ/ZBL).
    *   Implement `ActiveSetSelector`: Logic to use D-Optimality to select the most important structures from the pool before training.
*   **Outcome**: The system can take labeled structures and produce a `potential.yace` file.

### CYCLE 05: Dynamics Engine (Basic MD)
*   **Goal**: Run standard MD simulations with the generated potential.
*   **Features**:
    *   Implement `LAMMPSDynamics`.
    *   Develop the `HybridPotentialGenerator`: a utility that writes the `pair_style hybrid/overlay` commands for LAMMPS, combining the `.yace` file with the ZBL/LJ baseline.
    *   Basic MD loop: NPT/NVT heating and equilibration protocols.
*   **Outcome**: The system can run a stable MD simulation using the learned potential + physics baseline.

### CYCLE 06: On-the-Fly (OTF) Learning Loop
*   **Goal**: Close the Active Learning loop.
*   **Features**:
    *   Implement the "Watchdog": Configure LAMMPS `fix halt` to stop when `compute pace gamma` exceeds a threshold.
    *   Implement `HaltHandler`: Logic to parse the LAMMPS dump file, extract the high-uncertainty snapshot.
    *   Implement `LocalCandidateGenerator`: Generate perturbed variations of the halted structure to probe the local PES curvature.
*   **Outcome**: The system autonomously halts MD upon detecting uncertainty and feeds new candidates back to the Oracle.

### CYCLE 07: Advanced Dynamics (kMC & Scale-up)
*   **Goal**: Extend time-scale capabilities with EON.
*   **Features**:
    *   Implement `EONDynamics` wrapper.
    *   Create the `pace_driver.py` script that acts as the interface between EON (C++) and the Python Pacemaker calculator.
    *   Implement the OTF logic within the kMC step (halting kMC if the saddle point search hits high uncertainty).
*   **Outcome**: The system can perform long-timescale simulations (Ordering/Diffusion) and learn from transition states.

### CYCLE 08: Validation & Full Automation
*   **Goal**: Quality Assurance and Final Polish.
*   **Features**:
    *   Implement `StandardValidator`: Phonon dispersion stability check, Elastic constants calculation, EOS curve fitting.
    *   Implement `ReportGenerator`: Generate HTML/Markdown summaries of the training progress.
    *   Final "End-to-End" Integration Test using the Fe/Pt on MgO scenario.
*   **Outcome**: A fully polished, production-ready system with comprehensive validation reports.

## 6. Test Strategy

Testing will be conducted at multiple levels to ensure reliability across the 8 cycles.

### Unit Testing (Pytest)
*   **Scope**: Every individual class and method (e.g., `Config.validate`, `Structure.from_ase`).
*   **Approach**:
    *   Mock external dependencies: Use `unittest.mock` to simulate `ase.io.read`, `subprocess.run` (for LAMMPS/QE), and file system operations.
    *   Strict Type Checking: Use `mypy` to enforce data types defined in Domain Models.
    *   Coverage: Aim for >90% code coverage for the Core logic.

### Integration Testing (Component Level)
*   **Scope**: Interaction between Python code and external binaries (LAMMPS, Pacemaker, QE).
*   **Approach**:
    *   **Mock Binaries**: Create dummy shell scripts (`mock_lammps`, `mock_qe`) that accept input files and produce pre-defined valid output files. This allows testing the *parsing* and *execution logic* without running heavy physics calculations.
    *   **Container Tests**: Verify that the code runs correctly inside the provided Docker environment where real binaries exist (for "Real Mode" testing).

### End-to-End (E2E) System Testing
*   **Scope**: The full Active Learning Cycle (Orchestrator loop).
*   **Approach**:
    *   **Toy System**: Use a very simple system (e.g., Lennard-Jones Argon or Aluminum) where DFT can be calculated in seconds (or mocked with an EMA/EMT calculator).
    *   **Cycle verification**:
        *   Cycle 1: Verify folder creation.
        *   Cycle 3: Verify "Oracle" correctly returns energies.
        *   Cycle 6: Verify MD actually halts when uncertainty is artificially injected (or threshold set to 0).
    *   **Resilience Testing**: Intentionally trigger errors (e.g., "Disk Full", "DFT Divergence") and verify the Orchestrator pauses/alerts rather than crashing.

### Validation Testing (Scientific Accuracy)
*   **Scope**: The physics produced by the system.
*   **Approach**:
    *   Run the `Validator` suite against known literature values for standard materials (e.g., Si, Fe, MgO).
    *   Check if Phonon spectra are stable (no imaginary frequencies for stable crystals).
    *   Check if EOS curves are smooth and convex.

### User Acceptance Testing (UAT)
*   **Scope**: The User Experience.
*   **Approach**:
    *   Execute the `tutorials/` notebooks.
    *   Verify that `config.yaml` is the ONLY file the user needs to edit.
    *   Verify that error messages are human-readable and actionable.
