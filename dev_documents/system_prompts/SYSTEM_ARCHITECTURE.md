# System Architecture Document

## 1. Summary

The **PYACEMAKER** project represents a paradigm shift in the development and deployment of Machine Learning Interatomic Potentials (MLIP). At its core, it is an automated, self-driving laboratory for atomic-scale physics, designed to bridge the gap between rigorous quantum mechanical calculations (Density Functional Theory - DFT) and large-scale molecular dynamics (MD) simulations. By leveraging the **Atomic Cluster Expansion (ACE)** formalism via the **Pacemaker** library, PYACEMAKER enables the creation of potentials that combine the accuracy of DFT with the computational efficiency required for complex materials simulations.

The primary motivation for this system is the "democratization" of high-fidelity atomistic modelling. Traditionally, constructing a robust MLIP was a craft requiring years of expertise in both physics and data science. Researchers had to manually curate training sets, tune hyperparameters, and painstakingly validate potentials against experimental data. This manual process is not only prone to human error but also inefficient, often resulting in potentials that fail catastrophically in "unseen" regions of phase space (the extrapolation problem).

PYACEMAKER solves these challenges through a **Zero-Config**, **Active Learning** workflow. A user simply provides a high-level intent (e.g., "I need a potential for the Fe-Pt system suitable for high-temperature deposition") via a single YAML configuration file. The system's **Orchestrator** then takes over, autonomously executing a closed-loop cycle of:
1.  **Exploration**: Intelligently sampling the atomic configuration space using adaptive policies (MD, MC, Genetic Algorithms) to find diverse and representative structures.
2.  **Labelling**: invoking a robust **Oracle** (DFT engine) to calculate precise energies and forces, with built-in self-correction for convergence failures.
3.  **Training**: Fitting the ACE potential to this growing dataset, utilizing advanced active set selection to minimize data redundancy.
4.  **Validation**: Rigorously testing the potential against physical constraints (phonons, elastic constants) and providing immediate feedback.

Crucially, the system incorporates a "Safety First" philosophy. It enforces physical robustness through **Core Repulsion** (using ZBL/LJ baselines) to prevent atomic overlap and utilizes real-time **Uncertainty Quantification** (via the extrapolation grade $\gamma$) to detect when the simulation enters unknown territory. When high uncertainty is detected, the system pauses, performs a "Halt & Diagnose" procedure to learn the local physics, and then resumes—effectively "learning on the fly."

Furthermore, PYACEMAKER is designed to scale. It supports seamless transition from local workstations for development and debugging to massive HPC clusters for production runs. The integration of **Adaptive Kinetic Monte Carlo (aKMC)** via **EON** extends its reach beyond the nanosecond timescales of MD to the seconds and hours required for diffusion and ordering phenomena, making it a truly comprehensive tool for materials discovery.

## 2. System Design Objectives

The design of PYACEMAKER is guided by four overarching objectives, which serve as the "North Star" for all architectural decisions:

### 2.1. Zero-Config Automation (The "iPhone" Moment for MLIP)
**Goal:** To reduce the cognitive load on the user to near zero for standard tasks.
**Constraint:** The system must infer reasonable defaults for thousands of underlying parameters (DFT cutoffs, mixing betas, ACE basis set sizes, MD time steps) based on minimal user input (elements, approximate conditions).
**Success Criteria:** A novice user should be able to clone the repository, edit `config.yaml` to specify "Fe, Pt", and run `python main.py` to produce a production-ready potential without writing a single line of Python code. All errors (e.g., SCF non-convergence) should be handled internally whenever possible.

### 2.2. Data Efficiency via Active Learning (Smart over Big Data)
**Goal:** To maximize the information content per DFT calculation.
**Constraint:** DFT calculations are expensive ($O(N^3)$). We cannot afford to label millions of random structures.
**Success Criteria:** The system should achieve "chemical accuracy" (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with a dataset size that is 10x smaller than random sampling. This is achieved by actively selecting structures with high uncertainty ($\gamma$) or unique geometric descriptors (D-optimality), ensuring every CPU-hour spent on DFT contributes significantly to potential improvement.

### 2.3. Physics-Informed Robustness (Safety Nets)
**Goal:** To eliminate "unphysical" behaviours common in purely data-driven models.
**Constraint:** Polynomials (ACE) can oscillate wildly in extrapolation regions.
**Success Criteria:**
1.  **Core Repulsion:** The potential must *never* allow atoms to fuse (overlap < 1.0 Å) even at extreme energies. This is enforced by a hard-coded ZBL/LJ baseline.
2.  **Asymptotic Stability:** The potential should decay to zero (or a physical baseline) at long distances.
3.  **Dynamic Stability:** The final potential must produce stable phonon spectra (no imaginary frequencies) for the equilibrium crystal structure.

### 2.4. Modular Extensibility (Future-Proofing)
**Goal:** To allow easy integration of new scientific methods.
**Constraint:** The field of MLIP is moving fast. New descriptors (MACE, NequIP) or sampling methods (Generative flows) may emerge.
**Success Criteria:** The architecture must use strict interfaces (Abstract Base Classes). Adding a new DFT code (e.g., ABINIT) or a new Dynamics Engine (e.g., ASE-MD instead of LAMMPS) should strictly require implementing a subclass, without touching the core Orchestrator logic.

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture, with the **Orchestrator** acting as the central hub that coordinates independent, loosely coupled modules.

### 3.1. Core Components

1.  **Orchestrator (`src/mlip_autopipec/orchestrator.py`)**:
    *   **Role:** The "Brain". It manages the state machine of the active learning cycle.
    *   **Responsibilities:** Reads configuration, initializes modules, manages the loop (Exploration -> Labelling -> Training -> Validation), handles global error recovery, and manages the file system (directories for each iteration).

2.  **Structure Generator (`src/mlip_autopipec/structure_generator/`)**:
    *   **Role:** The "Explorer".
    *   **Responsibilities:** Generates candidate atomic structures. It implements "Adaptive Exploration Policies" to switch between Random Sampling, Heuristic Mutation (rattling, scaling), MD-based sampling, and Evolutionary Algorithms based on the current learning stage.

3.  **Oracle (`src/mlip_autopipec/oracle/`)**:
    *   **Role:** The "Truth Teller".
    *   **Responsibilities:** Wraps the DFT engine (Quantum Espresso, VASP). It handles input file generation, execution, output parsing, and crucial **Self-Correction** (e.g., reducing mixing beta if SCF diverges). It also performs **Periodic Embedding** to cut small, representative clusters from large MD snapshots for efficient calculation.

4.  **Trainer (`src/mlip_autopipec/trainer/`)**:
    *   **Role:** The "Learner".
    *   **Responsibilities:** Wraps the `Pacemaker` library. It manages the training dataset (serializing to `.pckl.gzip`), configures the fitting hyperparameters, runs the fitting process, and utilizes `pace_activeset` to select the most informative structures for the next generation.

5.  **Dynamics Engine (`src/mlip_autopipec/dynamics/`)**:
    *   **Role:** The "Prover".
    *   **Responsibilities:** Runs simulations (MD via LAMMPS, kMC via EON) using the current potential. It implements the **Hybrid Potential** (ACE + ZBL) logic and monitors the **Uncertainty Metric ($\gamma$)** in real-time. If $\gamma$ exceeds a threshold, it triggers a `Halt`, dumping the problematic structure for the Orchestrator to "heal".

6.  **Validator (`src/mlip_autopipec/validator/`)**:
    *   **Role:** The "Judge".
    *   **Responsibilities:** Runs a suite of physical tests (Phonons, Elastic Constants, EOS, Melting Point) to assess the quality of the potential beyond simple test-set errors.

### 3.2. Data Flow Diagram

```mermaid
graph TD
    User((User)) -->|config.yaml| Orch[Orchestrator]

    subgraph "Cycle: Active Learning Loop"
        Orch -->|1. Request Candidates| SG[Structure Generator]
        SG -->|2. Structures| Orch

        Orch -->|3. Submit Candidates| Oracle[Oracle (DFT)]
        Oracle -->|4. Labelled Data (E, F, S)| DB[(Training Dataset)]

        DB -->|5. Load Data| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. Fit Potential| Pot[potential.yace]

        Pot -->|7. Update| DE[Dynamics Engine (MD/kMC)]
        DE -->|8. Run Sim & Monitor| DE
        DE -- "Halt! (High Uncertainty)" --> SG
    end

    Pot -->|9. Validate| Val[Validator]
    Val -->|Pass| Prod[Production Ready]
    Val -->|Fail| Orch
```

## 4. Design Architecture

### 4.1. File Structure (ASCII Tree)

The project enforces a clean separation between source code, documentation, and operational data.

```
PYACEMAKER/
├── pyproject.toml              # Dependencies & Tool Config (Ruff, Mypy)
├── README.md                   # Entry point documentation
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py             # CLI Entry Point
│       ├── config/             # Pydantic Configuration Models
│       │   ├── __init__.py
│       │   └── main_config.py
│       ├── domain_models/      # Core Data Structures (DDD)
│       │   ├── structure.py    # Atoms, Cell, Constraints
│       │   └── potential.py    # Potential Objects, YACE wrappers
│       ├── interfaces/         # Abstract Base Classes
│       │   ├── abstract_oracle.py
│       │   ├── abstract_trainer.py
│       │   └── ...
│       ├── orchestrator/       # Core Logic
│       ├── oracle/             # DFT Implementations
│       ├── trainer/            # Pacemaker Wrappers
│       ├── dynamics/           # LAMMPS/EON Wrappers
│       ├── structure_generator/# Sampling Strategies
│       └── utils/              # Logging, IO Helpers
├── tests/                      # Pytest Suite
│   ├── unit/
│   └── integration/
├── dev_documents/              # Requirements & Specs
│   ├── ALL_SPEC.md
│   ├── SYSTEM_ARCHITECTURE.md
│   └── system_prompts/         # Cycle-specific Prompts
└── tutorials/                  # Jupyter Notebooks (UAT)
```

### 4.2. Domain Models (Pydantic)

We utilize **Pydantic** for rigorous data validation.

*   **`GlobalConfig`**: The root configuration object, validated against the YAML input. It contains nested models for `OracleConfig`, `TrainerConfig`, etc.
*   **`StructureMetadata`**: Captures provenance of an atomic structure (e.g., "Generated by RandomPerturbation from Structure #42").
*   **`ExplorationState`**: Tracks the current state of the active learning loop (Iteration #, Best Error, Current $\gamma$ threshold).
*   **`ValidationResult`**: A structured object containing pass/fail status and metrics for each validation test (Phonon, Elastic, etc.).

### 4.3. Key Design Patterns

*   **Factory Pattern**: Used to instantiate specific implementations (e.g., `OracleFactory.get_oracle("quantum_espresso")`) based on configuration.
*   **Strategy Pattern**: Used in the Structure Generator to switch between exploration strategies (Random vs MD vs Genetic) dynamically.
*   **Observer Pattern**: The Orchestrator subscribes to events from the Dynamics Engine (e.g., "Halt Triggered") to react immediately.

## 5. Implementation Plan

The project will be executed in **6 strict cycles**, each delivering a usable, tested increment of functionality.

### Cycle 01: Foundation & Orchestrator (Mocks)
*   **Goal:** Establish the "Skeleton" of the system.
*   **Features:**
    *   Setup `pyproject.toml`, `src` structure, and CI/CD basics.
    *   Implement the `Orchestrator` main loop and State Machine.
    *   Define all Abstract Base Classes (`BaseOracle`, `BaseTrainer`, etc.).
    *   Implement **Mock Components** (`MockOracle`, `MockTrainer`) that simulate delays and return dummy data.
    *   **Success:** A user can run `python main.py config.yaml` and see the logs of a full "fake" learning cycle completing successfully.

### Cycle 02: Oracle (DFT) & Structure Generator
*   **Goal:** Enable real data generation.
*   **Features:**
    *   Implement `StructureGenerator` with "Random Perturbation" and "Lattice Enumeration" strategies.
    *   Implement `Oracle` using **ASE (Atomic Simulation Environment)**.
    *   Support **Quantum Espresso** as the first backend.
    *   Implement **Periodic Embedding**: Logic to cut a cluster from a large cell and wrap it in a new periodic box with vacuum/buffer.
    *   **Success:** The system can take a crystal structure, perturb it, and run a real DFT calculation to get energy and forces.

### Cycle 03: Trainer Integration (Pacemaker)
*   **Goal:** Close the loop with actual ML training.
*   **Features:**
    *   Implement `PacemakerTrainer` wrapper.
    *   Handle data conversion: `ASE Atoms` -> `.pckl.gzip` (Pacemaker format).
    *   Implement `Active Set Selection` (using `pace_activeset`) to filter redundant data.
    *   Implement **Delta Learning**: Configure Pacemaker to learn $E_{total} - E_{ZBL}$.
    *   **Success:** The system can train a valid `.yace` potential file from the data generated in Cycle 02.

### Cycle 04: Dynamics Engine (MD) & Uncertainty
*   **Goal:** Enable "Active" learning via MD.
*   **Features:**
    *   Implement `DynamicsEngine` using **LAMMPS**.
    *   Implement **Hybrid Pair Style**: Automatically generate LAMMPS input for `pair_style hybrid/overlay pace zbl`.
    *   Implement **Uncertainty Watchdog**: Use `compute pace ... gamma_mode=1` and `fix halt` to stop MD when $\gamma > \text{threshold}$.
    *   Implement the "Halt & Diagnose" feedback loop to the Orchestrator.
    *   **Success:** An MD simulation runs and automatically stops when it encounters an unknown structure, which is then added to the training set.

### Cycle 05: Scale-Up (kMC & Adaptive Policy)
*   **Goal:** Extend to long timescales and smart exploration.
*   **Features:**
    *   Implement **Adaptive Exploration Policy**: A logic engine that decides *how* to sample (e.g., "If energy is low, do kMC; if high, do MD").
    *   Integrate **EON** for Adaptive Kinetic Monte Carlo (aKMC).
    *   Implement the interface between EON and the `.yace` potential (Python driver for EON).
    *   **Success:** The system can autonomously discover reaction pathways (e.g., diffusion events) that would be impossible to see with standard MD.

### Cycle 06: Validation & Production Readiness
*   **Goal:** Ensure scientific validity and user experience.
*   **Features:**
    *   Implement `Validator` suite: Phonon dispersion (via Phonopy), Elastic Constants, Equation of State (EOS).
    *   Implement the **Final UAT Scenarios** (Fe/Pt on MgO) as Jupyter Notebooks.
    *   Finalize `README.md`, documentation, and error messages.
    *   **Success:** The "Fe/Pt on MgO" tutorial runs end-to-end, producing a scientifically valid potential and reproducing known experimental properties.

## 6. Test Strategy

Testing is continuous and multi-layered.

### Cycle 01 Testing
*   **Unit**: Test config parsing and CLI argument handling.
*   **Integration**: Verify that `Orchestrator` correctly calls the `Mock` methods in the correct order.
*   **Verification**: Ensure the "Dry Run" completes without crashing and produces a log file.

### Cycle 02 Testing
*   **Unit**: Test `StructureGenerator` algorithms (e.g., ensure perturbations are within bounds).
*   **Integration**: Test `Oracle` with a "Mock DFT" (calculating LJ potential instead of real DFT) to save time, then one real QE calculation.
*   **Physics**: Verify `Periodic Embedding` preserves local geometric environments.

### Cycle 03 Testing
*   **Unit**: Test data serialization (ASE <-> Pacemaker).
*   **Integration**: Train a dummy potential on a small dataset (e.g., Lennard-Jones data) and verify it reproduces the training data.
*   **Performance**: Measure training time and memory usage.

### Cycle 04 Testing
*   **Unit**: Test LAMMPS input file generation (regex checks).
*   **Integration**: Run a short LAMMPS MD with a `fix halt` condition forced to trigger (low threshold) and verify the callback to Python works.
*   **Physics**: Verify the Hybrid Potential prevents atomic overlap (run two atoms at each other).

### Cycle 05 Testing
*   **Unit**: Test Policy logic (e.g., "Given state X, Policy returns Action Y").
*   **Integration**: Run a short EON kMC step using the potential.
*   **System**: Run a "Mini-Campaign" (Active Learning loop) for 1 hour to see if it discovers new structures autonomously.

### Cycle 06 Testing
*   **UAT**: Execute the `tutorials/*.ipynb` notebooks.
*   **Scientific Validation**:
    *   **Phonons**: Confirm Si bulk has no imaginary modes.
    *   **Elasticity**: Confirm C11, C12 values match reference.
    *   **EOS**: Confirm bulk modulus matches reference.
*   **Production**: "Monkey Test" - feed garbage config and ensure graceful exit.
