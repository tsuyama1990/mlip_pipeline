# System Architecture: PYACEMAKER

## 1. Summary

**PYACEMAKER** is a cutting-edge, automated software system designed to construct and operate Machine Learning Interatomic Potentials (MLIP) with high efficiency and reliability. At its core, it utilizes the **Atomic Cluster Expansion (ACE)** framework, provided by the "Pacemaker" engine, to generate state-of-the-art potentials. The primary mission of this project is to democratize the creation of high-quality MLIPs, making them accessible to materials scientists and researchers who may not possess deep expertise in data science or computational physics. By automating the complex and often tedious workflows associated with potential generation, PYACEMAKER enables users to focus on materials discovery rather than the intricacies of machine learning.

The system addresses several critical challenges inherent in traditional MLIP construction. Firstly, it tackles the issue of **structure sampling bias**. Standard Molecular Dynamics (MD) simulations often fail to visit "rare events" or high-energy configurations that are crucial for a robust potential. PYACEMAKER employs advanced exploration strategies, including adaptive policies and active learning, to systematically explore the configuration space, ensuring that the potential is trained on a diverse and representative dataset. This approach mitigates the risk of "extrapolation," where the potential behaves unpredictably in unknown regions.

Secondly, the system emphasizes **Data Efficiency**. Traditional methods often rely on brute-force generation of vast datasets, leading to the accumulation of "garbage" data—structures that are physically redundant and do not contribute to the potential's accuracy. PYACEMAKER utilizes **Active Learning** principles and **D-Optimality** criteria (via Active Set Optimization) to select only the most informative structures for First-Principles (DFT) calculation. This strategy aims to achieve high accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with significantly fewer expensive DFT calculations, reducing computational costs by an order of magnitude.

Thirdly, the system guarantees **Physics-Informed Robustness**. A common failure mode in MLIPs is the "hole" in the potential energy surface, where atoms can overlap without penalty, causing simulations to crash (segmentation faults). PYACEMAKER enforces physical correctness by combining the ML potential with a robust physics-based baseline (Lennard-Jones or ZBL). This hybrid approach ensures that even in data-sparse regions, the physics of atomic repulsion is preserved, preventing simulation instability.

Finally, PYACEMAKER is designed for **Scalability and Zero-Configuration**. Users can initiate the entire pipeline—from initial structure generation to final potential validation—using a single YAML configuration file. The architecture is modular and container-ready (Docker/Singularity), allowing seamless deployment from local workstations to high-performance computing (HPC) clusters. The system orchestrates the interaction between various specialized modules: the "Structure Generator" (Explorer), the "Oracle" (DFT Calculator), the "Trainer" (Pacemaker), the "Dynamics Engine" (LAMMPS/EON), and the "Validator" (Quality Assurance). This holistic approach transforms MLIP construction from a manual art into a repeatable, automated industrial process.

## 2. System Design Objectives

The design of PYACEMAKER is guided by a set of ambitious but achievable objectives, constraints, and success criteria. These goals shape every architectural decision and implementation detail.

### 2.1 Goals

1.  **Zero-Config Workflow (Automation)**:
    The primary goal is to minimize human intervention. The system must be capable of running the entire active learning loop—exploration, labelling, training, and validation—autonomously. A user should only need to provide a composition (e.g., "TiO2") and a few high-level constraints in a simple `config.yaml` file. The system must infer necessary hyperparameters, such as simulation temperatures or mixing ratios for exploration, using an "Adaptive Exploration Policy."

2.  **Data Efficiency (Cost Reduction)**:
    DFT calculations are the most expensive resource in this pipeline. The system aims to maximize the "Information Gain per CPU-hour." By employing uncertainty quantification (extrapolation grade $\gamma$) and active set optimization, the system filters out redundant structures. The target is to achieve production-quality accuracy with 1/10th of the training data required by random sampling methods.

3.  **Physics-Informed Robustness (Stability)**:
    The system must produce potentials that are safe to run. It is unacceptable for a production simulation to crash due to a "hole" in the potential. The design mandates the use of `pair_style hybrid/overlay` in LAMMPS, strictly enforcing a ZBL or LJ core repulsion. This ensures that the potential behaves physically (strong repulsion) at short interatomic distances, even if the ML model extrapolates poorly.

4.  **Scalability (HPC Readiness)**:
    The architecture must support parallel execution. While the orchestrator runs as a single process, the heavy lifting (DFT, MD, Training) must be dispatchable to MPI-enabled environments or batch job schedulers (Slurm/PBS). The design should not assume a shared file system for all processes, although a shared workspace is the initial target.

### 2.2 Constraints

*   **External Dependencies**: The system relies on third-party engines: Quantum Espresso (for DFT), LAMMPS (for MD), and Pacemaker (for ACE). The design must treat these as "black boxes" where possible but integrate tightly via input/output files and command-line interfaces.
*   **Computational Resources**: The system must be mindful of memory and disk usage. Trajectory files can be large; the system should stream data or subsample effectively rather than loading entire trajectories into RAM.
*   **Time Constraints**: The active learning loop involves iterative cycles. The "overhead" of the Python orchestrator must be negligible compared to the physics calculations.

### 2.3 Success Metrics

*   **Accuracy**: The final potential must satisfy RMSE thresholds: Energy < 1 meV/atom, Force < 0.05 eV/Å.
*   **Stability**: The potential must pass a phonon stability test (no imaginary frequencies) and satisfy Born stability criteria for elastic constants.
*   **Efficiency**: The "Time-to-Solution" (wall-clock time to generate a valid potential) should be minimized through parallel task execution and smart sampling.
*   **Usability**: A novice user must be able to install the package and run the "Quickstart" tutorial successfully without encountering obscure Python errors.

## 3. System Architecture

The architecture of PYACEMAKER follows a **Modular Orchestration Pattern**. A central "Orchestrator" manages the state and flow of data between specialized, loosely coupled modules.

### 3.1 Components

1.  **Orchestrator (The Brain)**:
    *   **Responsibility**: Manages the Active Learning Cycle. It maintains the `WorkflowState`, decides which phase to execute next, handles errors, and manages the file system structure.
    *   **Interaction**: Calls the APIs of other modules and reads their output.

2.  **Structure Generator (The Explorer)**:
    *   **Responsibility**: Proposes new atomic configurations.
    *   **Logic**: Uses an `AdaptivePolicy` to decide *how* to explore (MD, MC, Strain, Defects). It generates inputs for LAMMPS or creates structures directly via Python (ASE).

3.  **Dynamics Engine (The Executor)**:
    *   **Responsibility**: Runs simulations to sample the potential energy surface.
    *   **Tools**: LAMMPS (for MD), EON (for kMC).
    *   **Feature**: Implements "On-the-Fly (OTF)" monitoring. It watches the uncertainty metric ($\gamma$) and halts execution if it exceeds a threshold, returning the "dangerous" structure for labelling.

4.  **Oracle (The Sage)**:
    *   **Responsibility**: Provides ground-truth labels (Energy, Forces, Stress) via DFT.
    *   **Tools**: Quantum Espresso (primary), VASP (optional).
    *   **Feature**: Includes a "Self-Healing" mechanism. If a DFT calculation fails (e.g., SCF convergence error), it automatically retries with adjusted parameters (mixing beta, smearing).

5.  **Trainer (The Learner)**:
    *   **Responsibility**: Fits the ACE potential to the data.
    *   **Tools**: Pacemaker.
    *   **Feature**: Manages the dataset, performs Active Set Optimization to select the best training points, and runs the fitting process.

6.  **Validator (The Gatekeeper)**:
    *   **Responsibility**: Certifies the potential.
    *   **Tests**: Phonon dispersion, Elastic constants, EOS curves.
    *   **Output**: Generates a comprehensive HTML report.

### 3.2 Data Flow & Interaction (Mermaid)

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Manage State| State[Workflow State]

    subgraph "Active Learning Cycle"
        Orch -->|1. Request Structures| Gen[Structure Generator]
        Gen -->|2. Candidates / Task| Dyn[Dynamics Engine]
        Dyn -->|3. Exploration & OTF Halt| HighUncert[High Uncertainty Structures]
        HighUncert -->|4. Select & Embed| Oracle[Oracle (DFT)]
        Oracle -->|5. Labelled Data| DB[Dataset]
        DB -->|6. Train| Trainer[Trainer (Pacemaker)]
        Trainer -->|7. New Potential| Pot[Potential.yace]
        Pot -->|8. Update| Dyn
    end

    Pot -->|9. Validate| Val[Validator]
    Val -->|10. Report| Report[Validation Report]
    Val -->|Feedback| Orch
```

## 4. Design Architecture

The system is designed with a strict separation of concerns, utilizing **Pydantic** for data validation and **Protocol** classes (Interfaces) for extensibility.

### 4.1 File Structure

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py                 # Entry point
│       ├── config/                 # Configuration Models
│       │   ├── __init__.py
│       │   └── config_model.py     # Pydantic schema
│       ├── domain_models/          # Core Domain Objects
│       │   ├── structure.py        # Structure & Candidates
│       │   ├── potential.py        # Potential Artifacts
│       │   └── workflow.py         # State Management
│       ├── orchestration/          # Cycle Logic
│       │   ├── orchestrator.py
│       │   └── state.py
│       ├── physics/                # Physics Logic
│       │   ├── structure_gen/      # Generator Module
│       │   │   ├── generator.py
│       │   │   └── policy.py
│       │   ├── oracle/             # DFT Module
│       │   │   ├── dft_manager.py
│       │   │   └── espresso.py
│       │   ├── dynamics/           # MD/kMC Module
│       │   │   ├── lammps_runner.py
│       │   │   └── eon_wrapper.py
│       │   └── training/           # Trainer Module
│       │       └── pacemaker.py
│       ├── validation/             # QA Module
│       │   ├── validator.py
│       │   ├── metrics.py
│       │   └── report_generator.py
│       └── utils/
│           └── file_ops.py
├── tests/
│   ├── unit/
│   └── integration/
├── dev_documents/
├── pyproject.toml
└── README.md
```

### 4.2 Key Data Models

1.  **`Config` (Pydantic Model)**:
    *   The single source of truth for all settings. Validates types, ranges, and file paths upon startup.
    *   Sub-models: `ProjectConfig`, `LammpsConfig`, `DFTConfig`, `TrainingConfig`, `ValidationConfig`.

2.  **`WorkflowState`**:
    *   Tracks the current iteration, status of the potential, and history of metrics. It allows the system to be stopped and resumed (idempotency).

3.  **`Candidate`**:
    *   Represents an atomic structure identified for labeling. Contains the `Atoms` object (ASE), metadata (why it was selected, e.g., "high gamma"), and its source (trajectory frame).

4.  **`Potential`**:
    *   Represents a versioned `.yace` file. Contains metadata about its lineage (parent potential, training data used, validation score).

## 5. Implementation Plan

The project will be executed in **6 distinct cycles**, each adding a layer of functionality to the system.

### CYCLE 01: Foundation & Basic Loop
*   **Objective**: Establish the core framework, configuration parsing, and a basic "skeleton" loop that can train a potential using a pre-existing dataset.
*   **Features**:
    *   **Project Structure**: Setup `src` layout, `pyproject.toml`, logging, and error handling.
    *   **Config System**: Implement `Config` Pydantic models to parse and validate user inputs.
    *   **Orchestrator Skeleton**: Create the `Orchestrator` class that can iterate through a loop counter.
    *   **Basic Trainer**: Implement `PacemakerWrapper` to run `pace_train`.
    *   **Data Management**: Basic handling of `.pckl.gzip` datasets (reading/writing).
    *   **Mock Interfaces**: Stubs for Oracle and Dynamics to allow the loop to "run" without external physics engines.
*   **Deliverable**: A system that reads a config, "pretends" to run MD/DFT, and effectively runs a Pacemaker training job.

### CYCLE 02: The Oracle (DFT Automation)
*   **Objective**: Implement the robust DFT execution engine. This is the "Sage" that provides ground truth.
*   **Features**:
    *   **DFT Manager**: A class to manage the submission and retrieval of DFT jobs.
    *   **Quantum Espresso Interface**: Integration with `ase.calculators.espresso`.
    *   **Input Auto-Generation**: Automatic selection of pseudopotentials (SSSP) and k-points based on system density (`kspacing`).
    *   **Self-Healing Logic**: Try-catch blocks that detect SCF convergence failures and retry with altered parameters (mixing beta, smearing).
    *   **Force & Stress**: Ensure `tprnfor` and `tstress` are handled correctly to extract training labels.
*   **Deliverable**: A module that takes a list of ASE Atoms and returns them with computed Energy, Forces, and Stress, handling errors gracefully.

### CYCLE 03: The Explorer (Structure Generation)
*   **Objective**: Implement the intelligent sampling logic. The "Explorer" that decides what to calculate.
*   **Features**:
    *   **Structure Gen Interface**: Define the protocol for generators.
    *   **Strategies**: Implement `RandomSlice`, `Strain`, and `Defect` generators using ASE.
    *   **Adaptive Policy Engine**: A logic block that decides which strategy to use based on the input material (e.g., Metal vs. Insulator logic).
    *   **Periodic Embedding**: Logic to cut out a cluster from a large MD box and wrap it in a periodic supercell for DFT.
*   **Deliverable**: A module that can accept a "parent" structure and output a list of diverse "candidate" structures for the Oracle.

### CYCLE 04: The Executor (Dynamics & Active Learning)
*   **Objective**: Connect the loop with real MD simulations and On-the-Fly (OTF) monitoring.
*   **Features**:
    *   **LAMMPS Runner**: A robust wrapper for the LAMMPS binary.
    *   **Hybrid Potential**: Implementation of `pair_style hybrid/overlay` generation to mix ACE with ZBL/LJ.
    *   **OTF Monitoring**: Implementation of `fix halt` logic in LAMMPS to stop when `v_max_gamma` exceeds the threshold.
    *   **Halt & Diagnose**: Orchestrator logic to parse LAMMPS logs, identify the halt step, and extract the problematic structure.
*   **Deliverable**: The full "Active Learning" cycle. MD runs -> Halts on uncertainty -> Structure extracted -> Sent to Oracle -> Retraining.

### CYCLE 05: The Validator (Quality Assurance)
*   **Objective**: Implement the testing suite to verify potential quality.
*   **Features**:
    *   **Validation Runner**: A manager to run post-training tests.
    *   **Phonon Test**: Integration with `phonopy` to calculate band structures and check for imaginary frequencies.
    *   **Elasticity Test**: Calculation of elastic tensor and Born stability checks.
    *   **EOS Test**: Equation of State fitting (Birch-Murnaghan).
    *   **Report Generator**: Creation of `validation_report.html` with plots (using `matplotlib` or `plotly`).
*   **Deliverable**: A system that outputs a "Pass/Fail" grade and a visual report for every generated potential.

### CYCLE 06: Advanced Integration (kMC & Production)
*   **Objective**: Extend the system to long-timescale phenomena and finalize for production.
*   **Features**:
    *   **EON Integration**: Wrapper for the EON software to run aKMC (Adaptive Kinetic Monte Carlo).
    *   **kMC-OTF Link**: Logic to handle uncertainty halts during saddle point searches in EON.
    *   **Active Set Optimization**: Full implementation of `pace_activeset` to prune datasets.
    *   **Final Polish**: CLI improvements, comprehensive error messages, and documentation.
*   **Deliverable**: The complete PYACEMAKER system, capable of handling both MD and kMC exploration, ready for public release.

## 6. Test Strategy

Testing is critical for a complex system involving external physics engines.

### 6.1 General Approach
*   **Pytest**: The primary test runner.
*   **Mocking**: Heavy use of `unittest.mock` to simulate external binaries (LAMMPS, QE, Pacemaker) during unit tests. We cannot rely on these binaries being present in the CI environment for all tests.
*   **CI/CD**: GitHub Actions to run the test suite on every commit.

### 6.2 Cycle-Specific Test Plans

*   **Cycle 01 (Foundation)**:
    *   **Unit**: Test Config parsing with valid/invalid YAMLs. Test basic file I/O.
    *   **Integration**: Run the Orchestrator with a "MockTrainer" that just copies a dummy potential. Verify the state machine transitions correctly.

*   **Cycle 02 (Oracle)**:
    *   **Unit**: Test input file generation for QE. Test parsing of QE output files (XML/text).
    *   **Integration**: If QE is installed, run a small calculation (e.g., single Si atom). If not, mock the process execution and verify the "Self-Healing" logic triggers when a specific error string is injected into the mock stderr.

*   **Cycle 03 (Generator)**:
    *   **Unit**: Verify `PeriodicEmbedding` logic ensures minimal distance between periodic images. Test that `DefectGenerator` correctly removes/adds atoms.
    *   **Visual**: Generate structures and save them to `.xyz` to visually inspect (manual check during dev).

*   **Cycle 04 (Dynamics)**:
    *   **Unit**: Check `in.lammps` generation for correct `pair_style hybrid` syntax.
    *   **Integration**: Run a short LAMMPS MD (using a dummy potential) and trigger a `fix halt` by manually forcing a high value in a variable (if possible) or providing a potential that is known to fail. Verify the Orchestrator catches the halt signal.

*   **Cycle 05 (Validator)**:
    *   **Unit**: Test Phonon parsing logic. Test Elasticity calculation math.
    *   **Integration**: Run validation on a known "Good" potential and a known "Bad" potential. Ensure the system correctly Flags the bad one (e.g., imaginary phonons).

*   **Cycle 06 (Production)**:
    *   **E2E (End-to-End)**: A full "mini" run. Start with 1 structure, run 1 cycle of exploration, labeling, training. This requires a "Real Mode" environment with all binaries.
    *   **Performance**: Measure the overhead of the Python orchestration logic.
