# System Architecture: PYACEMAKER

## 1. Summary

The PYACEMAKER project represents a paradigm shift in the computational materials science domain, specifically addressing the complexities associated with constructing and deploying Machine Learning Interatomic Potentials (MLIPs). Traditionally, the development of high-fidelity potentials, such as those based on the Atomic Cluster Expansion (ACE) formalism, has been the purview of domain experts possessing deep knowledge in both quantum mechanics (Density Functional Theory - DFT) and data science. This exclusivity has created a significant bottleneck, preventing the wider adoption of MLIPs in industrial R&D and academic exploration where rapid material screening is critical.

PYACEMAKER is designed as an "Autopilot for Atomistic Simulation," a comprehensive system that automates the entire lifecycle of an MLIP. From the initial generation of atomic structures to the final deployment of a robust potential for molecular dynamics (MD) or kinetic Monte Carlo (kMC) simulations, the system operates with minimal human intervention. The core philosophy is "Zero-Config," where a user provides a simple high-level intent (e.g., "I want a potential for Fe-Pt alloys") via a single YAML configuration file, and the system handles the intricate orchestration of data generation, model training, validation, and refinement.

The system is built upon a modular, microservices-inspired architecture where distinct components—the **Structure Generator**, **Oracle**, **Trainer**, and **Dynamics Engine**—collaborate under the strict supervision of a central **Orchestrator**. This separation of concerns ensures scalability and maintainability. For instance, the Oracle module, responsible for expensive DFT calculations, is designed to be self-healing, automatically adjusting convergence parameters when calculations fail, a common pain point in high-throughput workflows. Similarly, the Dynamics Engine does not merely run simulations but actively monitors the "uncertainty" of the potential in real-time, halting execution before physical laws are violated (e.g., atomic overlap) and triggering a retraining loop.

Furthermore, PYACEMAKER emphasizes "Physics-Informed" robustness. It avoids the black-box nature of pure ML models by enforcing physical baselines (such as Lennard-Jones or ZBL potentials) for short-range interactions. This ensures that even in data-sparse regions, the simulation remains physically plausible, preventing catastrophic failures common in pure polynomial fits. The integration of active learning strategies, specifically D-Optimality based selection, ensures that the system learns efficiently, selecting only the most informative atomic configurations for labeling, thereby reducing the computational cost of DFT by an order of magnitude compared to random sampling.

In summary, PYACEMAKER is not just a wrapper around existing tools; it is an intelligent, autonomous agent that democratizes access to state-of-the-art atomistic modeling, enabling researchers to focus on materials discovery rather than the nuances of potential fitting.

## 2. System Design Objectives

The design of PYACEMAKER is guided by four primary objectives, each addressing specific challenges in the current landscape of atomistic simulation. These objectives serve as the compass for all architectural decisions and implementation details.

### 2.1. Zero-Config Workflow & Democratization
The foremost objective is to lower the barrier to entry. Current workflows often require users to write complex Python scripts, manage file paths manually, and tune dozens of hyperparameters for DFT and ML models. PYACEMAKER aims to eliminate this "accidental complexity."
- **Goal**: A user should be able to start a production-grade training run with a configuration file of less than 50 lines.
- **Mechanism**: We employ "Sensible Defaults" and "Adaptive Logic." For example, instead of asking the user for a k-point grid, the system calculates it based on the cell size and a target density. Instead of manually specifying temperature schedules for exploration, the system infers reasonable ranges from the material's melting point (estimated via coarse-grained models if unknown).
- **Success Metric**: A novice user with basic knowledge of Linux should be able to install the package and produce a validated potential for a binary alloy within 24 hours of computing time, without writing a single line of code.

### 2.2. Data Efficiency via Active Learning
DFT calculations are computationally expensive and carbon-intensive. A brute-force approach that samples thousands of random structures is inefficient and often fails to capture rare but critical events.
- **Goal**: Achieve "State-of-the-Art" accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with 1/10th the data of traditional methods.
- **Mechanism**: The system implements a rigorous Active Learning loop. The **Dynamics Engine** continuously monitors the extrapolation grade ($\gamma$) of the potential. When high uncertainty is detected, the simulation pauses, and the problematic structure is not just labeled, but used as a seed to generate a "physically relevant" local dataset (via normal mode perturbation or short MD bursts). We then use Linear D-Optimality (MaxVol algorithm) to select the mathematically optimal subset of structures that maximizes the information gain for the ACE basis set.
- **Impact**: This minimizes the number of expensive DFT calls, directing resources only to where the model is ignorant, rather than re-learning known regions.

### 2.3. Physics-Informed Robustness
Machine learning models are interpolators. When forced to extrapolate (e.g., during high-energy collisions in radiation damage simulations), they often behave unphysically.
- **Goal**: Guarantee simulation stability and physical correctness in the "far-from-equilibrium" regime.
- **Mechanism**: We enforce a "Delta Learning" architecture. The ML model (ACE) learns the *difference* between the true DFT energy and a robust physical baseline (Lennard-Jones or ZBL).
- **Constraint**: The system strictly enforces the presence of a short-range repulsive wall. Even if the ML model predicts a non-physical attraction at very short distances, the ZBL baseline dominates, preventing atoms from fusing. This "Safety First" approach allows the system to run unsupervised without crashing due to numerical instabilities.

### 2.4. Scalability & Extensibility
Scientific requirements evolve. A system built today for bulk crystals must support complex interfaces, defects, or multi-component alloys tomorrow.
- **Goal**: A modular architecture that supports "Plug-and-Play" components and scales from a laptop to an HPC cluster.
- **Mechanism**: The use of Abstract Base Classes (ABCs) for all core components (`BaseOracle`, `BaseTrainer`, etc.) allows for easy substitution. For instance, replacing Quantum Espresso with VASP is a matter of writing a new adapter class, not rewriting the orchestrator.
- **Scale**: The **Orchestrator** is designed to handle asynchronous job submission (future proofing for Slurm/PBS), and the data pipeline uses serialized formats (Pickle/Gzip) that handle datasets efficiently. The integration with EON (kMC) demonstrates the system's ability to span time scales, extending its utility beyond nanosecond MD to second-scale diffusion phenomena.

## 3. System Architecture

The system follows a centralized orchestration pattern where the `Orchestrator` acts as the conductor, and other components act as specialized musicians.

### Component Overview
1.  **Orchestrator**: The brain. It manages the global state (current cycle, best potential), handles the control flow (when to explore, when to train), and ensures fault tolerance (recovering from crashes).
2.  **Structure Generator**: The explorer. Responsible for proposing new atomic configurations. It uses "Policies" to decide whether to run high-temperature MD, insert defects, or apply strain, based on the current uncertainty landscape.
3.  **Oracle**: The ground truth provider. Wraps DFT codes (Quantum Espresso) to calculate energy, forces, and stresses. It includes a "Self-Healing" layer that automatically fixes common convergence errors.
4.  **Trainer**: The learner. Wraps the `pacemaker` library. It manages the training dataset, performs active set selection (D-Optimality), and fits the ACE potential.
5.  **Dynamics Engine**: The validator and stress-tester. Runs MD and kMC simulations using the current potential. It features an "On-the-Fly" (OTF) watchdog that halts execution if the extrapolation grade exceeds a threshold.
6.  **Validator**: The auditor. Runs a battery of physical tests (Phonons, Elastic constants, EOS) to certify the potential's quality before release.

### Data Flow Diagram

```mermaid
graph TD
    subgraph "Control Plane"
        Orch[Orchestrator]
        Config[Global Config (YAML)]
    end

    subgraph "Data Plane"
        DS[Dataset (Pickle/Gzip)]
        Pot[Potential (YACE)]
    end

    subgraph "Compute Plane"
        SG[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        DE[Dynamics Engine (MD/kMC)]
        Val[Validator]
    end

    Config --> Orch
    Orch --> SG
    Orch --> Oracle
    Orch --> Trainer
    Orch --> DE
    Orch --> Val

    %% Active Learning Loop
    DE -- "1. High Uncertainty / Novelty" --> SG
    SG -- "2. Candidate Structures" --> Oracle
    Oracle -- "3. Labeled Data (E, F, S)" --> DS
    DS --> Trainer
    Trainer -- "4. New Potential" --> Pot
    Pot --> DE
    Pot --> Val
    Val -- "5. Validation Report" --> Orch
```

### Interaction Details
The "Active Learning Cycle" is the heartbeat of the system.
1.  **Exploration**: The `DynamicsEngine` runs a simulation (e.g., MD at 1000K).
2.  **Detection**: The OTF watchdog calculates $\gamma$ for every atom. If $\gamma > \gamma_{thresh}$, the simulation halts.
3.  **Selection**: The halted structure is passed to the `StructureGenerator`, which creates local perturbations (candidates) to probe the unknown direction.
4.  **Labeling**: The `Oracle` computes the exact forces for these candidates. To save cost, it uses "Periodic Embedding" to cut out a small cluster around the uncertain region, reducing the DFT system size.
5.  **Training**: The `Trainer` adds the new data to the `Dataset`, re-optimizes the Active Set, and updates the `Potential`.
6.  **Deployment**: The new potential is hot-swapped into the `DynamicsEngine`, and the simulation resumes.

## 4. Design Architecture

This section details the software design, focusing on the Python implementation, file structure, and data models.

### 4.1. File Structure (ASCII Tree)

The project structure enforces separation of concerns and facilitates packaging.

```
mlip-pipeline/
├── pyproject.toml              # Dependency and build management
├── README.md                   # Entry point documentation
├── config.yaml                 # Default configuration template
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py             # CLI Entry point
│       ├── domain_models/      # Pydantic models (Data Transfer Objects)
│       │   ├── __init__.py
│       │   ├── config.py       # GlobalConfig, ComponentConfigs
│       │   ├── structure.py    # Structure, Dataset definitions
│       │   └── potential.py    # Potential, ExplorationResult
│       ├── interfaces/         # Abstract Base Classes
│       │   ├── __init__.py
│       │   ├── oracle.py
│       │   ├── trainer.py
│       │   ├── dynamics.py
│       │   └── generator.py
│       ├── infrastructure/     # Concrete Implementations
│       │   ├── __init__.py
│       │   ├── oracle/         # QE/VASP adapters
│       │   ├── trainer/        # Pacemaker wrapper
│       │   ├── dynamics/       # LAMMPS/EON wrappers
│       │   └── generator/      # Structure manipulation logic
│       ├── orchestrator/       # Logic wiring
│       │   ├── __init__.py
│       │   └── simple_orchestrator.py
│       └── utils/              # Helpers
│           ├── __init__.py
│           ├── logging.py
│           └── physics.py      # Unit conversions, constants
├── tests/                      # Pytest suite
│   ├── unit/
│   └── integration/
├── dev_documents/              # Documentation
└── tutorials/                  # Jupyter Notebooks
```

### 4.2. Data Models (Pydantic)

We use Pydantic V2 for strict type validation and configuration management.

**Global Configuration (`config.py`)**:
The single source of truth. It validates that the user inputs (e.g., temperatures, cutoffs) are physically meaningful.
- `ProjectConfig`: Paths, names, seed.
- `OracleConfig`: DFT parameters (k-spacing, smearing, pseudopotentials).
- `TrainerConfig`: Pacemaker parameters (basis size, cutoffs, loss weights).
- `DynamicsConfig`: MD settings (timestep, thermostat, OTF thresholds).

**Structure Object (`structure.py`)**:
A wrapper around `ase.Atoms` but with enforced metadata.
- Ensures presence of Energy, Forces, and Stress for labeled data.
- Tracks provenance (which iteration generated this structure?).

**Exploration Result (`potential.py`)**:
The output from the Dynamics Engine.
- Contains the `final_structure`, `trajectory_path`, `termination_reason` (e.g., "MaxSteps" or "UncertaintyHalt"), and the `max_gamma` observed.

## 5. Implementation Plan

The development is divided into 6 distinct cycles to manage complexity and risk.

### Cycle 01: System Skeleton & Mock Infrastructure
**Goal**: Establish the "Walking Skeleton" – a pipeline that runs end-to-end with fake data.
- **Features**:
    - CLI entry point (`typer`) parsing the YAML config.
    - `Orchestrator` logic implementing the main loop.
    - Abstract Base Classes (`BaseOracle`, `BaseTrainer`, etc.).
    - **Mock Implementations**: `MockOracle` (returns random forces), `MockTrainer` (creates dummy file), `MockDynamics` (simulates random halting).
- **Deliverable**: A system that reads config and "pretends" to train a potential for 10 cycles, verifying the orchestration logic.

### Cycle 02: Data Generation & DFT Integration
**Goal**: Replace the `MockOracle` and `MockStructureGenerator` with real physics.
- **Features**:
    - `StructureGenerator`: Implement `RandomDisplacement` and `Supercell` creation.
    - `Oracle`: Implement `DFTManager` using ASE's `Espresso` calculator.
    - **Self-Healing**: Implement the `try-except` blocks for DFT convergence (mixing beta reduction).
    - **Periodic Embedding**: Logic to cut out clusters from bulk structures for efficient labeling.
- **Deliverable**: Ability to generate a real dataset from DFT calculations for simple bulk structures.

### Cycle 03: Machine Learning Core
**Goal**: Integrate `Pacemaker` for actual training.
- **Features**:
    - `Trainer`: Wrapper around `pace_train`.
    - `ActiveSet`: Integration of `pace_activeset` (MaxVol) to select representative structures.
    - **Delta Learning**: Setup for `pair_style hybrid/overlay` (ZBL + ACE).
    - `Dataset`: Management of `.pckl.gzip` files (merging, splitting train/test).
- **Deliverable**: A pipeline that can train a working ACE potential from a static dataset.

### Cycle 04: Dynamics Engine & OTF Learning
**Goal**: Close the Active Learning loop with real MD.
- **Features**:
    - `DynamicsEngine`: Interface to `LAMMPS` (via `lammps` python module).
    - **OTF Watchdog**: Implementation of `fix halt` based on `compute pace` gamma values.
    - **Hybrid Potential Generator**: Automatic generation of `in.lammps` with correct `pair_style hybrid`.
    - **Recovery Logic**: Parsing LAMMPS logs to identify the "Halted Structure."
- **Deliverable**: The full "Exploration -> Halt -> Retrain" cycle working for a simple melt-quench scenario.

### Cycle 05: Advanced Exploration & kMC
**Goal**: Expand the domain to long time scales and complex sampling.
- **Features**:
    - **Adaptive Policy**: Logic to switch between High-T MD and Strain sampling based on material type.
    - `EONWrapper`: Interface to run `eonclient` for kMC saddle point searches.
    - **Driver Script**: Python script for EON to call the ACE potential.
    - **Defect Generation**: Strategies to introduce vacancies and interstitials.
- **Deliverable**: Ability to simulate diffusion and ordering phenomena (Fe/Pt on MgO case).

### Cycle 06: Validation, Robustness & Production
**Goal**: Ensure the potential is trustworthy and the system is user-friendly.
- **Features**:
    - `Validator`: Automated Phonon and Elasticity calculation (calling `phonopy` and simple shear deformations).
    - **Reporting**: Generation of HTML/Markdown reports with plots.
    - **Tutorials**: Jupyter Notebooks for the User Acceptance Test.
    - **Error Handling**: Graceful shutdowns and state saving (checkpointing).
- **Deliverable**: v1.0 Release Candidate.

## 6. Test Strategy

Testing is continuous and multi-layered.

### Unit Testing
Every Python module in `src/` must have corresponding tests in `tests/unit/`.
- **Framework**: `pytest`.
- **Coverage**: Aim for >80% code coverage.
- **Mocks**: Heavy use of `unittest.mock` to avoid calling actual DFT codes or LAMMPS during unit tests. We test the *logic* of the wrapper, not the external binary.
- **Property-Based Testing**: Use `hypothesis` (optional) to generate random configurations and ensure parsers don't crash.

### Integration Testing
Tests that verify component interactions.
- **Mock Mode Integration**: A dedicated test suite that runs the full `Cycle 01` mock pipeline. This ensures that changes in interfaces don't break the orchestration.
- **Tool Integration**:
    - **DFT**: A test that runs a tiny SCF calculation on silicon (1 atom) if `pw.x` is present in the environment.
    - **LAMMPS**: A test that runs a 10-step MD on LJ argon to verify the LAMMPS python binding.
    - **Pacemaker**: A test that trains a potential on 5 structures to verify `pace_train` invocation.

### System/End-to-End Testing (UAT)
- **CI Pipeline**: On every Pull Request, the "Mock Mode" pipeline must complete successfully.
- **Nightly Builds**: If resources allow, a "Real Mode" run on a small system (e.g., Al) to verify physics convergence.
- **Documentation Tests**: All code snippets in README and Notebooks must be executable.

### Specific Strategy per Cycle
- **Cycle 01**: Focus on Config validation and Orchestrator state transitions.
- **Cycle 02**: Verify DFT input generation (files match expected format) and error parsing regex.
- **Cycle 03**: Verify Tensor shapes of the dataset and that `pace_train` produces a valid `.yace` file.
- **Cycle 04**: Verify LAMMPS `fix halt` triggers correctly by artificially injecting a high gamma value.
- **Cycle 05**: Verify EON communication protocol (stdin/stdout) works with the driver script.
- **Cycle 06**: Verify Validator correctly identifies "Bad" potentials (e.g., by manually creating a potential with negative bulk modulus).
