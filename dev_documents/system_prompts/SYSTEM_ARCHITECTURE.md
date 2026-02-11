# System Architecture: PYACEMAKER

## 1. Summary

The PYACEMAKER (High Efficiency MLIP Construction & Operation System) represents a paradigm shift in the computational materials science domain, specifically addressing the "democratisation" of Machine Learning Interatomic Potentials (MLIP). Traditionally, the construction of high-fidelity potentials, such as those based on the Atomic Cluster Expansion (ACE) formalism, has been the purview of domain experts possessing a rare combination of skills in quantum mechanics, statistical physics, and data science. The manual curation of training datasets, the delicate tuning of density functional theory (DFT) parameters, and the iterative refinement of potential parameters constitute a formidable barrier to entry for experimentalists and industrial researchers. PYACEMAKER dismantles this barrier by providing a fully automated, "Zero-Config" workflow that transforms a simple material specification into a production-ready, validated interatomic potential.

At its core, the system is designed to be an autonomous research agent. It does not merely execute a linear script; it observes, reasons, and acts. Through a sophisticated "Orchestrator" architecture, the system manages a cyclical workflow known as the Active Learning Cycle. This cycle begins with **Exploration**, where a Structure Generator employs an Adaptive Exploration Policy to navigate the vast chemical and structural space of the target material. Unlike traditional random sampling, this module uses physics-informed heuristics—such as phase stability predictions and uncertainty quantification—to propose candidate structures that are most likely to improve the potential's accuracy.

The system then moves to **Labelling**, where the "Oracle" module (wrapping Quantum Espresso or VASP) performs high-fidelity DFT calculations. Crucially, this Oracle is equipped with self-healing capabilities. It can detect common convergence failures, diagnose the underlying cause (e.g., charge sloshing, magnetic instability), and automatically adjust calculation parameters (e.g., mixing beta, smearing width) to recover the calculation without user intervention. This resilience is vital for high-throughput autonomous operations.

Following labelling, the **Trainer** module utilises the Pacemaker engine to fit the ACE potential. It employs advanced data selection techniques, such as D-Optimality (MaxVol algorithm), to curate a "sparse but information-rich" training set, preventing data redundancy and minimising computational costs. The trainer also enforces physical robustness by implementing Delta Learning against a robust baseline (Lennard-Jones or ZBL), ensuring that the potential behaves physically even in high-energy collision regimes where training data is absent.

Finally, the **Dynamics Engine** deploys the trained potential into molecular dynamics (MD) and kinetic Monte Carlo (kMC) simulations. This engine is not a passive runner; it actively monitors the "Extrapolation Grade" ($\gamma$) of the simulation in real-time. If the simulation wanders into a region of the potential energy surface where the model is uncertain, the engine triggers a "Halt," captures the problematic configuration, and feeds it back into the learning loop. This On-the-Fly (OTF) learning capability ensures that the potential becomes progressively smarter and more robust the longer it is used.

The system culminates in a **Validator** module that acts as a strict Quality Gate. It rigorously tests the generated potential against physical laws—checking for phonon stability, elastic tensor positive definiteness (Born criteria), and equation of state smoothness—before certifying it for production use. By integrating these complex components into a seamless, robust, and user-friendly platform, PYACEMAKER empowers researchers to focus on materials discovery rather than method development.

## 2. System Design Objectives

The architectural design of PYACEMAKER is driven by five foundational objectives, each serving as a pillar to ensure the system's utility, reliability, and longevity.

### 2.1. Democratisation via Zero-Config Usability
The primary objective is to lower the barrier to entry for MLIP construction. The system must operate on the principle of "Convention over Configuration." A user should be able to initiate a complex active learning campaign with a single YAML configuration file defining only the most essential material properties (e.g., "Ti-O system"). All technical hyperparameters—DFT cutoffs, k-point grids, ACE basis set sizes, active learning thresholds—must have robust, physics-based defaults that adapt dynamically to the system at hand. The user experience should be akin to using a modern appliance: complex on the inside, simple on the outside.

### 2.2. Data Efficiency through Active Learning
Computational resources, particularly DFT calculations, are expensive and finite. A naive approach of "more data is better" is unsustainable. The system aims to maximise the information gain per unit of computational cost. This is achieved through two mechanisms:
1.  **Uncertainty-Driven Sampling:** The system prioritises calculations on structures where the current potential is least confident, effectively "exploring the unknown" rather than "confirming the known."
2.  **D-Optimality Selection:** Even among candidate structures, the system filters out redundant configurations using linear algebra techniques (determinant maximisation) to select a basis set that spans the widest possible variance in the descriptor space. The goal is to achieve State-of-the-Art (SOTA) accuracy (RMSE Energy < 1 meV/atom) with 1/10th of the training data required by random sampling methods.

### 2.3. Physical Robustness and Safety
Machine learning models are notorious for behaving unpredictably outside their training domain (extrapolation). In materials science, this can lead to catastrophic simulation failures (e.g., atoms overlapping, resulting in infinite forces). The system must guarantee "Physical Safety." This is enforced by:
*   **Hybrid Potentials:** Always coupling the ML potential with a physics-based baseline (ZBL/LJ) to handle short-range repulsion.
*   **Strict Validation:** The inclusion of a dedicated Validator module that checks fundamental physical laws (phonons, elasticity) ensures that the potential is not just fitting data points but capturing the underlying physics of the material. A potential that fits energy errors perfectly but predicts unstable phonons for a stable crystal is rejected.

### 2.4. Autonomous Resilience (Self-Healing)
For a system to run unsupervised for days or weeks, it must be resilient to failure. The architecture treats errors not as exceptions but as expected states. The Oracle module's ability to "self-heal" DFT calculations is a prime example. If a calculation crashes, the system should not crash; it should analyse the error log, apply a fix (e.g., change algorithm, reduce mixing), and retry. This objective extends to the entire pipeline: if a job hangs, it should be timed out and killed; if a file is corrupt, it should be detected. The goal is a system that requires zero "babysitting."

### 2.5. Scalability across Time and Space
Materials phenomena occur across vast scales. The system architecture must support:
*   **Spatial Scalability:** From small unit cells in DFT to million-atom MD simulations in LAMMPS.
*   **Temporal Scalability:** Bridging the gap between the femtosecond scale of MD and the second/hour scale of diffusion and aging. This is achieved by integrating Adaptive Kinetic Monte Carlo (aKMC) via EON. The architecture must treat MD and kMC as interchangeable "Exploration Drivers," feeding the same central potential and learning from the unique configurations generated by both distinct sampling methods.

## 3. System Architecture

The PYACEMAKER system is architected as a modular, event-driven pipeline orchestrated by a central controller. The components are designed to be loosely coupled, interacting through well-defined file-based interfaces and configuration objects, allowing for independent development and testing of each module.

### 3.1. High-Level Component Diagram

```mermaid
graph TD
    %% Actors
    User([User])
    HPC[HPC Scheduler / Local Workstation]

    %% Core System
    subgraph PYACEMAKER [PYACEMAKER System]
        style PYACEMAKER fill:#f9f9f9,stroke:#333,stroke-width:2px

        %% Orchestrator
        Orchestrator[Orchestrator<br/>(The Brain)]
        style Orchestrator fill:#ffcccc,stroke:#333,stroke-width:2px

        %% Modules
        Generator[Structure Generator<br/>(The Explorer)]
        Oracle[Oracle / DFT<br/>(The Judge)]
        Trainer[Trainer / Pacemaker<br/>(The Learner)]
        Dynamics[Dynamics Engine<br/>(The Runner)]
        Validator[Validator<br/>(The Gatekeeper)]

        %% Data Stores
        StateDB[(State Store<br/>JSON/Pickle)]
        PotRepo[(Potential<br/>Repository)]
        DataLake[(Training<br/>Data Lake)]

        %% Flows
        User -->|config.yaml| Orchestrator
        Orchestrator -->|Manage| StateDB

        %% Cycle Flow
        Orchestrator -->|1. Request Candidates| Generator
        Generator -->|Candidates| Orchestrator

        Orchestrator -->|2. Request Labels| Oracle
        Oracle -->|Labeled Data| DataLake

        Orchestrator -->|3. Train| Trainer
        DataLake --> Trainer
        Trainer -->|potential.yace| PotRepo

        Orchestrator -->|4. Deploy & Run| Dynamics
        PotRepo --> Dynamics
        Dynamics -->|Uncertainty Halt| Generator
        Dynamics -->|Performance Metrics| Orchestrator

        Orchestrator -->|5. Verify| Validator
        PotRepo --> Validator
        Validator -->|Validation Report| Orchestrator
    end

    %% External Interactions
    Oracle -.->|Submit Jobs| HPC
    Dynamics -.->|Run LAMMPS/EON| HPC
```

### 3.2. Component Interaction & Data Flow

1.  **Orchestrator:** The central nervous system. It reads the `config.yaml`, initializes the workflow state, and dispenses tasks to other modules. It tracks the current cycle number, manages the directory structure for each iteration, and decides when to transition from exploration to training, or when to terminate the active learning loop based on convergence criteria.

2.  **Structure Generator:** Responsible for proposing new atomic configurations. In the early stages (Cold Start), it uses heuristics or pre-trained universal potentials (M3GNet) to guess stable structures. In later stages (Active Learning), it receives "Halted" structures from the Dynamics Engine—configurations where the current potential failed—and generates local perturbations (candidates) around them to probe the unknown energy landscape.

3.  **Oracle:** The interface to ground-truth physics (DFT). It accepts candidate structures and returns their energy, forces, and virial stresses. It encapsulates the complexity of DFT codes (Quantum Espresso), handling input file generation, pseudopotential selection (SSSP), and crucially, error recovery. It also handles "Periodic Embedding," cutting out clusters from large MD snapshots and wrapping them in vacuum-padded boxes for isolated DFT calculation.

4.  **Trainer:** Wraps the Pacemaker library. It takes the accumulated labelled data from the Data Lake and fits the ACE potential. It manages the "Active Set," selecting the most informative structures to keep the regression problem tractable. It also enforces the "Delta Learning" protocol, ensuring the ACE model learns only the correction to the baseline ZBL/LJ potential.

5.  **Dynamics Engine:** The runtime environment. It runs LAMMPS for MD and EON for kMC. It implements the "Hybrid/Overlay" pair style to combine ACE with the physical baseline. It runs the "Watchdog" (fix halt), monitoring the extrapolation grade $\gamma$ at every step. If $\gamma$ exceeds a threshold, it kills the simulation and returns the failing snapshot to the Orchestrator, triggering a retraining cycle.

6.  **Validator:** The quality assurance module. After every training cycle, it runs a battery of physical tests—Phonon Dispersion, Elastic Constants, Equation of State—on the new potential. It provides a Go/No-Go signal to the Orchestrator, preventing a physically broken potential from being used in the next rigorous simulation phase.

## 4. Design Architecture

The software design follows a strict Object-Oriented approach reinforced by Pydantic for data validation and schema definition. This ensures that all internal data structures are strongly typed and self-documenting.

### 4.1. File Structure (ASCII Tree)

```text
pyacemaker/
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py                     # CLI Entry Point
│       ├── core/
│       │   ├── orchestrator.py         # Main Workflow Manager
│       │   ├── state_manager.py        # Persistence & Resume Logic
│       │   └── logger.py               # Central Logging
│       ├── domain_models/
│       │   ├── config.py               # Global Config Pydantic Models
│       │   ├── datastructures.py       # Atoms, Trajectories, Potentials
│       │   └── enums.py                # Status Codes, Calculation Types
│       ├── components/
│       │   ├── generator/
│       │   │   ├── base.py
│       │   │   ├── adaptive_policy.py  # Exploration Strategy Logic
│       │   │   └── structure_gen.py
│       │   ├── oracle/
│       │   │   ├── base.py
│       │   │   ├── qe_driver.py        # Quantum Espresso Interface
│       │   │   └── self_healer.py      # Error Recovery Logic
│       │   ├── trainer/
│       │   │   ├── base.py
│       │   │   └── pacemaker.py        # Pacemaker Wrapper
│       │   ├── dynamics/
│       │   │   ├── base.py
│       │   │   ├── lammps_driver.py
│       │   │   └── eon_driver.py       # kMC Interface
│       │   └── validator/
│       │       ├── base.py
│       │       └── physics_check.py    # Phonons, Elasticity
│       └── utils/
│           ├── shell.py                # Subprocess Management
│           └── io.py                   # File Helpers
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
└── README.md
```

### 4.2. Key Data Models

*   **`GlobalConfig`**: A monolithic Pydantic model representing the `config.yaml`. It is composed of sub-models (`OrchestratorConfig`, `DFTConfig`, `TrainingConfig`, etc.). It performs validation at startup (e.g., checking if required executables paths are valid).
*   **`WorkflowState`**: A serialisable object (JSON) that tracks the progress of the active learning campaign. It stores the current iteration index, the paths to the latest potential and dataset, and the status of the current step (e.g., `EXPLORATION_DONE`, `TRAINING_FAILED`). This allows the system to be stopped and resumed at any point.
*   **`StructureMetadata`**: An extension of the standard Atoms object (ASE), attaching vital MLIP-context metadata such as `provenance` (where did this come from? MD halt? Random gen?), `uncertainty_score` (gamma value), and `status` (Labels pending, Labels computed, Failed).

## 5. Implementation Plan

The project is decomposed into 8 sequential cycles. Each cycle builds upon the previous one, adding a layer of functionality until the full autonomous loop is realised.

### Cycle 01: Core Framework & Orchestrator Skeleton
*   **Objective:** Establish the project foundation and the central control logic.
*   **Features:**
    *   Setup `pyproject.toml`, directory structure, and CI/CD pipelines.
    *   Implement `GlobalConfig` using Pydantic to parse and validate user inputs.
    *   Implement `Orchestrator` class with a "Mock" loop (cycling through states without doing real work).
    *   Implement `StateManager` for save/resume functionality.
    *   Setup centralised logging.
*   **Outcome:** A runnable CLI application (`mlip-runner`) that reads a config, sets up directories, and logs the flow of a hypothetical simulation.

### Cycle 02: Structure Generator (Exploration)
*   **Objective:** Implement the "Explorer" capability to generate training candidates.
*   **Features:**
    *   Implement `StructureGenerator` interface.
    *   Develop `AdaptiveExplorationPolicy`: Logic to decide *how* to sample (MD vs MC, Temperature Ramps) based on material properties.
    *   Integrate `M3GNet` / `CHGNet` (optional/mockable) for "Cold Start" initial structure generation.
    *   Implement algorithms for random distortion and supercell creation.
*   **Outcome:** The system can now generate a diverse set of atomic structures (XYZ files) ready for calculation.

### Cycle 03: Oracle (DFT Automation & Self-Healing)
*   **Objective:** Implement the "Judge" capability to compute ground-truth labels.
*   **Features:**
    *   Implement `Oracle` interface and `QEDriver` (Quantum Espresso).
    *   Implement `InputGenerator`: Auto-selection of Pseudopotentials (SSSP) and K-points.
    *   **Crucial:** Implement `SelfHealer`. A finite-state machine that catches QE crashes, adjusts parameters (mixing beta, smearing), and retries.
    *   Implement `PeriodicEmbedding`: Logic to cut a cluster from a large box and wrap it in a small periodic box for DFT.
*   **Outcome:** The system can take a list of structures and reliably return a list of Labeled Atoms (Energy/Forces/Virials), robust to calculation failures.

### Cycle 04: Trainer (Pacemaker & Active Set)
*   **Objective:** Implement the "Learner" capability to build the potential.
*   **Features:**
    *   Implement `Trainer` interface and `PacemakerWrapper`.
    *   Implement `ActiveSetSelector`: Integration with `pace_activeset` (MaxVol) to filter data.
    *   Implement `DeltaLearning`: Logic to subtract ZBL/LJ baseline energy before training.
    *   Manage the `train.pckl.gzip` dataset file handling (merging new data).
*   **Outcome:** The system can take labeled structures and produce a `potential.yace` file that minimizes the validation error.

### Cycle 05: Dynamics Engine I (LAMMPS MD & Uncertainty)
*   **Objective:** Implement the "Runner" capability for Molecular Dynamics.
*   **Features:**
    *   Implement `DynamicsEngine` interface and `LammpsDriver`.
    *   Implement `HybridPotentialBuilder`: Auto-generate LAMMPS input scripts using `pair_style hybrid/overlay` (ACE + ZBL).
    *   Implement `UncertaintyWatchdog`: Configure `fix halt` in LAMMPS to stop simulation when `gamma > threshold`.
*   **Outcome:** The system can run MD simulations that automatically stop when they encounter unknown physics, returning the "dangerous" structure.

### Cycle 06: OTF Loop Integration
*   **Objective:** Close the loop. Connect all components into the "Brain."
*   **Features:**
    *   Wire the Orchestrator to handle the full cycle:
        1.  Run Dynamics -> Catch Halt.
        2.  Extract Halt Structure -> Generate Candidates (Cycle 02).
        3.  Compute Labels (Cycle 03).
        4.  Re-train Potential (Cycle 04).
        5.  Restart Dynamics.
    *   Implement convergence checks (e.g., "Stop if MD runs for 1ns without halts").
*   **Outcome:** A fully autonomous Active Learning loop that improves the potential iteratively.

### Cycle 07: Dynamics Engine II (EON & aKMC)
*   **Objective:** Expand the "Runner" to long time scales using Kinetic Monte Carlo.
*   **Features:**
    *   Implement `EONDriver`: Interface to the EON software suite.
    *   Implement `PythonPotDriver`: A bridge script allowing EON to call the Python `PacemakerCalculator` for energy/forces.
    *   Integrate Uncertainty Check in aKMC: If a saddle point search hits high uncertainty, trigger the OTF loop.
*   **Outcome:** The system can simulate diffusion and ordering phenomena that are impossible with MD alone.

### Cycle 08: Validator & Reporting
*   **Objective:** Implement the "Gatekeeper" and reporting.
*   **Features:**
    *   Implement `Validator` module.
    *   Integrate `Phonopy` for phonon stability checks.
    *   Implement Elastic Constant calculation (Born criteria).
    *   Implement Equation of State (EOS) checks.
    *   Generate `report.html`: A dashboard showing learning curves, parity plots, and validation pass/fail status.
*   **Outcome:** The system provides a guarantee of physical correctness and a human-readable summary of the campaign.

## 6. Test Strategy

Testing is integral to the development of a scientific instrument like PYACEMAKER. We employ a pyramid testing strategy, ensuring correctness from individual functions up to the full scientific workflow.

### Cycle 01 Test Strategy
*   **Unit Tests:** Verify `GlobalConfig` validation rules (e.g., "error if work_dir exists"). Test `StateManager` save/load consistency.
*   **Integration Tests:** Run the `Orchestrator` in "Mock Mode." Ensure it creates the correct directory tree (`iter_001`, `iter_002`) and transitions through states without crashing.
*   **CLI Tests:** Verify `mlip-runner --help` and basic command invocation.

### Cycle 02 Test Strategy
*   **Unit Tests:** Verify `AdaptiveExplorationPolicy` logic (e.g., "If material is metallic, propose High-MC policy").
*   **Integration Tests:** Feed a dummy structure to `StructureGenerator` and verify it outputs valid XYZ files with perturbed coordinates. Check that `M3GNet` wrapper returns a reasonable structure (or a mock structure if CI).

### Cycle 03 Test Strategy
*   **Unit Tests:** Verify `InputGenerator` creates correct QE input strings (checking flags like `tprnfor=.true.`). Verify `PeriodicEmbedding` math (does the box size match the cut radius?).
*   **Integration Tests:** "Mock Oracle" test. Simulate a QE crash (return non-zero exit code) and verify `SelfHealer` generates a new input file with lower mixing beta. Finally, simulate success and check if parsing logic extracts correct forces.

### Cycle 04 Test Strategy
*   **Unit Tests:** Verify data merging logic (no duplicate structures). Verify `DeltaLearning` math (Energy_ACE = Energy_Total - Energy_ZBL).
*   **Integration Tests:** Run `PacemakerWrapper` against a tiny mock dataset. Verify it produces a `.yace` file and that the `active_set` command reduces the dataset size.

### Cycle 05 Test Strategy
*   **Unit Tests:** Verify LAMMPS input script generation (check `hybrid/overlay` syntax).
*   **Integration Tests:** Run `LammpsDriver` with a dummy potential. Force a "Halt" event (by artificially injecting a high gamma value or using a broken potential) and verify the driver catches the `fix halt` exit code and returns the correct snapshot frame.

### Cycle 06 Test Strategy
*   **System Tests:** The "Mini-Loop" test. Initialize with a random potential. Run the full Orchestrator. Assert that:
    1.  MD runs and halts.
    2.  Oracle "computes" (mocked) the halt structure.
    3.  Trainer produces a "new" potential (mocked timestamp update).
    4.  The loop increments the iteration counter.

### Cycle 07 Test Strategy
*   **Unit Tests:** Verify `EON` config file generation.
*   **Integration Tests:** Verify the `PythonPotDriver` interface. Can EON call our Python script and get energy/forces? Test the uncertainty interrupt in the bridge script.

### Cycle 08 Test Strategy
*   **Unit Tests:** Verify `PhononCheck` logic (does it detect imaginary frequencies in a known unstable list?). Verify `BornCriteria` math.
*   **Integration Tests:** Run the Validator against a "Good" potential (Lennard-Jones) and a "Bad" potential (Random noise). Verify it passes the first and fails the second with the correct error message. Verify HTML report generation.
