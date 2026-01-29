# System Architecture: PyAceMaker

## 1. Summary

The **PyAceMaker** project represents a significant leap forward in the domain of computational materials science, specifically addressing the challenges associated with constructing and maintaining Machine Learning Interatomic Potentials (MLIPs). Historically, the creation of high-quality MLIPs has been a craft reserved for a select few—researchers possessing a rare combination of deep expertise in quantum mechanics (Density Functional Theory or DFT), statistical mechanics (Molecular Dynamics or MD), and modern data science. The barrier to entry is prohibitively high, often involving manual, error-prone workflows where structures are hand-picked, calculations are manually monitored for convergence errors, and potential fitting parameters are tuned via trial and error.

This system aims to **democratise atomistic simulation** by providing a fully automated, "Zero-Config" pipeline. At its core, it leverages the **Atomic Cluster Expansion (ACE)** formalism via the `pacemaker` engine, which offers a mathematically rigorous and computationally efficient way to represent the potential energy surface. Unlike neural networks which can be opaque "black boxes", ACE provides a systematic basis set expansion that is both interpretable and incredibly fast.

However, the power of a potential is only as good as the data it is trained on. A critical flaw in many existing MLIP workflows is the reliance on standard equilibrium Molecular Dynamics for training data generation. This approach fails to capture "rare events"—such as bond breaking, diffusion barriers, or high-energy collisions—which are precisely the phenomena of interest in many applications. When a simulation encounters a configuration outside its training set (extrapolation), standard potentials often behave unphysically, predicting attractive forces where atoms should repel, leading to simulation crashes ("exploding atoms").

PyAceMaker addresses this via a robust **Active Learning** strategy. It does not merely learn from a static dataset; it actively explores the chemical and structural space. The system employs an **Adaptive Exploration Policy** that intelligently switches between Molecular Dynamics (MD) and Monte Carlo (MC) sampling, or applies specific strain and defect engineering strategies, depending on the material's characteristics (e.g., metals vs. insulators).

Furthermore, the system incorporates a **Self-Healing Oracle**. DFT calculations are notorious for failing due to convergence issues. This system wraps the Quantum Espresso engine with a recovery logic that automatically adjusts mixing parameters, smearing widths, or algorithms to salvage calculations that would otherwise require human intervention.

Finally, the system ensures **Physics-Informed Robustness**. It enforces a physical baseline (Lennard-Jones or ZBL potential) for short-range interactions. This "Delta Learning" approach ensures that even in deep extrapolation regions where the ML model might be uncertain, the fundamental laws of physics (Pauli exclusion principle/core repulsion) prevent the simulation from catastrophic failure.

In essence, PyAceMaker is an autonomous robotic researcher. It formulates hypotheses (generates structures), conducts experiments (runs DFT), learns from results (trains ACE), and verifies its own knowledge (validates via phonons/elasticity), iterating until it produces a potential that is not just accurate, but robust and ready for production-scale simulations.

## 2. System Design Objectives

The design of PyAceMaker is guided by a set of stringent objectives and constraints, ensuring that the final product is not only functional but also scalable, maintainable, and user-friendly.

### 2.1. Zero-Config Workflow (Automation First)
The primary objective is to minimise human time. A user should be able to define a material system (e.g., "Ti-O binary system") and a desired accuracy in a single configuration file, and the system should handle the rest.
-   **Goal**: From `init` to `production_potential.yace` without manual intervention.
-   **Metric**: The system must automatically handle DFT convergence errors, selection of hyperparameters for exploration, and training regularisation.
-   **Constraint**: All parameters must have sensible physical defaults derived from the atomic species (e.g., atomic radii, masses).

### 2.2. Data Efficiency (Smart Sampling)
DFT calculations are expensive (computationally O(N^3)). We cannot afford to compute thousands of redundant structures.
-   **Goal**: Achieve "State-of-the-Art" accuracy (RMSE < 1 meV/atom) with 1/10th the data of random sampling.
-   **Strategy**: Utilise **D-Optimality** (Active Set Selection) to select only the most mathematically distinct structures for the training set.
-   **Strategy**: Use **Periodic Embedding** to cut small, representative clusters from large MD simulations, calculating forces only for the local environment of interest, thereby saving computational resources.

### 2.3. Physical Robustness & Safety
A potential that is accurate 99% of the time but crashes the simulation 1% of the time is useless for long-term runs.
-   **Goal**: Absolute stability in MD simulations.
-   **Strategy**: Implement **Delta Learning**. The total energy is $E_{total} = E_{baseline} + E_{ML}$. The baseline (ZBL/LJ) handles the physics of overlapping atoms, while the ML model learns the complex chemistry.
-   **Strategy**: Real-time **Uncertainty Quantification**. The system monitors the extrapolation grade ($\gamma$) during MD. If $\gamma$ exceeds a threshold, the simulation halts *before* it crashes, triggering an immediate learning cycle.

### 2.4. Scalability and Extensibility
The architecture must support future expansions without rewriting the core.
-   **Goal**: Modular design.
-   **Requirement**: The "Dynamics Engine" must abstract the underlying solver, allowing us to swap LAMMPS (MD) for EON (Kinetic Monte Carlo) seamlessly.
-   **Requirement**: The "Oracle" must be calculator-agnostic, supporting Quantum Espresso now but VASP or CASTEP in the future via ASE interfaces.

### 2.5. Comprehensive Validation
Trust is earned through rigorous testing.
-   **Goal**: Automated Quality Assurance.
-   **Requirement**: Every generated potential must pass a battery of physical tests: Phonon dispersion stability (no imaginary modes), Elastic tensor consistency (Born stability criteria), and EOS curve smoothness.
-   **Output**: An HTML report ("The Certificate") must be generated for each cycle.

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture, with the **Orchestrator** acting as the central brain. It coordinates the data flow between four specialised "Worker" modules.

### 3.1. Components

1.  **The Orchestrator**:
    -   The central controller that manages the workflow state.
    -   Persists state to disk (supporting resume-on-failure).
    -   Decides the next phase based on the current cycle and validation results.

2.  **Structure Generator (The Explorer)**:
    -   Responsible for proposing new atomic configurations.
    -   **Modes**:
        -   *Cold Start*: Uses heuristic random packing or external databases (M3GNet) to get initial seed structures.
        -   *Adaptive Exploration*: Uses an active policy to decide whether to run high-temperature MD, pressure ramps, or Monte Carlo swaps based on the material type (e.g., metal vs. insulator).

3.  **Dynamics Engine (The Verifier & Miner)**:
    -   Runs the actual simulations using the current potential.
    -   **Role**: To stress-test the potential and find "holes" in the knowledge.
    -   **Mechanism**: Runs LAMMPS MD with a `fix halt` command linked to the extrapolation grade ($\gamma$). If the simulation enters an unknown region, it halts and returns the "failed" structure to the Orchestrator.

4.  **Oracle (The Teacher)**:
    -   Provides the "Ground Truth".
    -   Wraps Quantum Espresso (QE).
    -   **Key Feature**: "Self-Healing". It parses QE output files for specific error patterns (e.g., "convergence not achieved") and automatically modifies input parameters (mixing beta, smearing) to retry.
    -   Performs **Periodic Embedding**: Takes a cluster of atoms from a halted MD run, wraps it in a buffer of vacuum-padded periodic box, and calculates forces.

5.  **Trainer (The Learner)**:
    -   Wraps `pacemaker`.
    -   Manages the training dataset (Active Set).
    -   Fits the ACE potential to the difference between DFT forces and Baseline (LJ/ZBL) forces.

6.  **Validator (The Auditor)**:
    -   Independent of the training loop.
    -   Calculates physical properties (Phonons, Elasticity, EOS).
    -   Decides if a potential is "Production Ready".

### 3.2. Architecture Diagram

```mermaid
graph TD
    User[User Configuration] -->|config.yaml| Orch[Orchestrator]

    subgraph "Core Logic"
        Orch -->|Request Structures| Gen[Structure Generator]
        Orch -->|Request Simulation| Dyn[Dynamics Engine]
        Orch -->|Request Ground Truth| Oracle[Oracle (DFT)]
        Orch -->|Request Training| Trainer[Trainer (Pacemaker)]
        Orch -->|Request Validation| Valid[Validator]
    end

    subgraph "Data Flow"
        Gen -->|Candidate Structures| Dyn
        Dyn -->|Halted/Uncertain Structures| Oracle
        Oracle -->|Labelled Data (E, F, S)| Trainer
        Trainer -->|Potential (.yace)| Dyn
        Trainer -->|Potential (.yace)| Valid
        Valid -->|Pass/Fail & Report| Orch
    end

    subgraph "External Tools"
        Dyn -.-> LAMMPS
        Dyn -.-> EON(kMC)
        Oracle -.-> QE[Quantum Espresso]
        Trainer -.-> Pacemaker
        Valid -.-> Phonopy
    end
```

## 4. Design Architecture

The system is designed using a strict **Schema-First** approach. All data flowing between modules is validated using Pydantic models. This ensures type safety and prevents "silent failures" where data is malformed but processing continues.

### 4.1. File Structure

```ascii
src/mlip_autopipec/
├── domain_models/          # Pydantic Schemas (The "Language" of the system)
│   ├── structure.py        # Atom, Structure, Supercell definitions
│   ├── potential.py        # Potential metadata, path, type
│   ├── calculation.py      # DFT inputs/outputs, convergence flags
│   └── config.py           # Global configuration schema
├── orchestration/          # The Brain
│   ├── workflow.py         # Main Loop, State Machine
│   ├── state_manager.py    # Persistence logic (pickle/json)
│   └── phases/             # Logic for each phase (Exploration, Training, etc.)
├── physics/                # Domain Logic
│   ├── structure_gen/      # Policies, Random packing, Defects
│   ├── dft/                # QE Wrapper, Input generation, Recovery
│   ├── dynamics/           # LAMMPS/EON wrappers, Log parsing
│   ├── training/           # Pacemaker wrapper, Active Set selection
│   └── validation/         # Phonons, Elasticity, EOS calculators
├── infrastructure/         # Technical Plumbing
│   ├── cli.py              # Typer/Click entry points
│   ├── io.py               # File handling, serialisation
│   └── logging.py          # Rich logging setup
└── app.py                  # Application Entry Point
```

### 4.2. Key Data Models

1.  **`WorkflowState`**:
    -   Tracks the current cycle number, the current phase (e.g., `EXPLORATION`, `TRAINING`), and the paths to the latest potential and dataset.
    -   Ensures the system can be stopped and resumed at any point.

2.  **`CandidateStructure`**:
    -   Represents a structure identified for calculation.
    -   Attributes: `ase_atoms` (the actual geometry), `origin` (e.g., "MD_halt_cycle_3"), `uncertainty_score` (the $\gamma$ value), `status` (PENDING, CALCULATING, DONE, FAILED).

3.  **`DFTResult`**:
    -   The output from the Oracle.
    -   Attributes: `energy`, `forces` (Nx3 array), `stress` (3x3 or Voigt), `converged` (bool), `metadata` (mixing parameters used).
    -   Strict validation ensures arrays match the number of atoms.

4.  **`ValidationReport`**:
    -   Aggregates results from sub-validators.
    -   Attributes: `rmse_energy`, `rmse_force`, `phonon_stable` (bool), `bulk_modulus_error` (float).

## 5. Implementation Plan

The project is decomposed into 8 distinct cycles, each building upon the previous one to evolve from a skeleton to a fully autonomous system.

### **Cycle 01: Foundation & Core Models**
-   **Goal**: Establish the "Language" of the system.
-   **Deliverables**:
    -   Project skeleton and directory structure.
    -   Pydantic models for `Config`, `Structure`, `DFTInput`, `DFTOutput`.
    -   Configuration loader (YAML to Pydantic).
    -   Centralised logging system.
    -   **Why**: Without a strict schema, later integrations will be messy.

### **Cycle 02: Basic Exploration (The "One-Shot" Pipeline)**
-   **Goal**: Get data moving.
-   **Deliverables**:
    -   `LammpsRunner`: A wrapper to execute LAMMPS.
    -   `Orchestrator` (v1): A linear script that runs one MD simulation.
    -   Basic `StructureBuilder`: Random perturbation of a crystal.
    -   **Why**: We need to verify we can drive external binaries (LAMMPS) from Python.

### **Cycle 03: The Oracle (DFT Automation)**
-   **Goal**: Reliable ground truth.
-   **Deliverables**:
    -   `QERunner`: Wrapper for `pw.x`.
    -   `InputGenerator`: Converts Pydantic models to QE input files (including pseudopotentials).
    -   `RecoveryHandler`: The logic to fix failed SCF calculations.
    -   **Why**: DFT is the bottleneck; it must be robust before we automate the loop.

### **Cycle 04: The Learner (Pacemaker Integration)**
-   **Goal**: Close the loop (Train a model).
-   **Deliverables**:
    -   `PacemakerRunner`: Wrapper for `pace_train`.
    -   `DatasetManager`: Converts ASE atoms to Pacemaker's pickle format.
    -   `ActiveSetSelector`: Integration of `pace_activeset`.
    -   **Why**: We need to actually produce a `.yace` file to proceed.

### **Cycle 05: Validation Framework**
-   **Goal**: Trust but verify.
-   **Deliverables**:
    -   `PhononValidator` (using `phonopy`).
    -   `ElasticityValidator`.
    -   `EOSValidator`.
    -   HTML Report Generator.
    -   **Why**: We need metrics to know if our training in Cycle 04 was successful.

### **Cycle 06: The Active Learning Loop**
-   **Goal**: Autonomy.
-   **Deliverables**:
    -   `Orchestrator` (v2): State-machine based loop.
    -   `UncertaintyMonitor`: Parsing LAMMPS logs for `fix halt` events.
    -   `StructureExtractor`: Cutting clusters from halted MD frames.
    -   **Why**: This connects C02, C03, and C04 into a continuous self-improving cycle.

### **Cycle 07: Adaptive Strategy**
-   **Goal**: Smarter exploration.
-   **Deliverables**:
    -   `AdaptivePolicy`: Logic to switch between MD and MC.
    -   `DefectGenerator`: Strategies to create vacancies/interstitials.
    -   `StrainGenerator`: Sampling deformed unit cells.
    -   **Why**: Random MD is inefficient; we need to target specific regions of phase space.

### **Cycle 08: Expansion & Production**
-   **Goal**: Real-world capability.
-   **Deliverables**:
    -   `EONWrapper`: Integration with Kinetic Monte Carlo.
    -   `ProductionDeployer`: Finalising the potential for distribution.
    -   Full System End-to-End Tests.
    -   **Why**: To support long-timescale phenomena (kMC) and ensure the system is ready for users.

## 6. Test Strategy

Testing is not an afterthought; it is integral to the development of a scientific instrument where accuracy is paramount.

### 6.1. Unit Testing
-   **Scope**: Individual functions and classes.
-   **Focus**:
    -   **Pydantic Models**: Ensure validation rules trigger correctly (e.g., rejecting a structure with mismatching positions and symbols).
    -   **Parsers**: Test log parsers (LAMMPS, QE) against mocked text output to ensure they correctly identify convergence, errors, and values.
    -   **Logic**: Test the `AdaptivePolicy` logic (e.g., "If material is metal, suggest MD").
-   **Tools**: `pytest`, `pytest-mock`.

### 6.2. Integration Testing
-   **Scope**: Interaction between modules.
-   **Focus**:
    -   **IO**: Verify that files written by `InputGenerator` are readable by the external binary (conceptually) or match expected string output.
    -   **Wrappers**: Mock the external binary execution (using `subprocess` mocks) to verify that the correct command-line arguments are constructed.
    -   **Data Flow**: Verify that a `Structure` object can be converted to a `DFTInput`, and a `DFTOutput` can be converted back to a `TrainingEntry`.

### 6.3. System/End-to-End Testing (Simulated)
-   **Scope**: Full cycles.
-   **Focus**:
    -   Since we cannot run heavy DFT/MD in the CI environment, we will use **"Mock Calculators"**.
    -   **Mock Oracle**: A Python function that accepts a structure and returns a fake energy (e.g., computed via a Lennard-Jones potential) instead of running QE.
    -   **Mock Trainer**: A dummy trainer that outputs a pre-existing `.yace` file.
    -   **Scenario**: Run the Orchestrator for 2 cycles using these mocks to ensure the file handling, state transitions, and error handling logic work flawlessly.

### 6.4. Scientific Validation (UAT)
-   **Scope**: The physics.
-   **Focus**:
    -   This is covered by the `Validator` module in Cycle 05.
    -   In UAT, we will run the full pipeline on a cheap system (e.g., Aluminum or Silicon) and verify that the resulting potential reproduces known properties (lattice constant, elastic moduli) within 10%.

### 6.5. Pre-commit Checks
-   **Tools**: `ruff` (linting), `mypy` (type checking).
-   **Policy**: No code shall be committed if it fails strict type checking. This enforces the schema-first design.
