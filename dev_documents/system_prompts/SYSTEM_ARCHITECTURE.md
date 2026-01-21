# System Architecture

## 1. Summary

The **MLIP-AutoPipe (Machine Learning Interatomic Potential - Automated Pipeline)** is an autonomous, scalable, and self-correcting software system designed to eliminate human intervention from the complex, iterative, and error-prone process of constructing machine learning potentials (MLPs).

### Background and Motivation
In the domain of computational materials science, the creation of accurate interatomic potentials is a significant bottleneck, often referred to as the "Chicken and Egg" problem. To generate high-quality training data, one requires robust molecular dynamics (MD) simulations that explore high-energy and high-temperature configurations. However, to run these simulations effectively without physical crashes, one first needs a reliable potential. This circular dependency typically necessitates tedious manual intervention, where researchers must:
1.  Hand-craft initial datasets using intuition or random sampling.
2.  Babysit Density Functional Theory (DFT) calculations, manually restarting them when they fail due to convergence issues.
3.  Manually fit potentials, tweaking hyperparameters.
4.  Cautiously test them in small steps, often restarting the whole process when the potential causes an MD simulation to explode.

This manual workflow is not only slow but also unscalable. It relies heavily on "tacit knowledge" (phronesis) that is hard to codify. MLIP-AutoPipe aims to codify this knowledge into a software artifact.

### The "Zero-Human" Protocol
MLIP-AutoPipe solves this by automating the entire lifecycle through a "Zero-Human" protocol. A user simply inputs a material composition (e.g., "Fe-Ni Alloy") and a high-level simulation goal (e.g., "Melt Quench" or "Phase Diagram"). The system then orchestrates a sophisticated, self-correcting workflow involving multiple external physics engines:
-   **Quantum Espresso**: For high-fidelity, first-principles DFT calculations. This provides the "Ground Truth" labels (Energy, Forces, Stress).
-   **LAMMPS**: For massive-scale MD simulations. This acts as the "Explorer", pushing the material into new thermodynamic states.
-   **Pacemaker**: For training Atomic Cluster Expansion (ACE) potentials. This acts as the "Student", learning from the DFT data.
-   **MACE**: For surrogate-based candidate screening. This acts as the "Scout", quickly identifying promising structures before expensive DFT is run.

### Key Architectural Innovations

#### 1. Surrogate-First Exploration
In traditional active learning, candidates for DFT are often chosen randomly or via simple heuristics. This is inefficient because DFT is computationally expensive ($O(N^3)$). MLIP-AutoPipe introduces a "Surrogate-First" layer. Before submitting any structure to DFT, the system employs pre-trained foundation models (like MACE-MP, trained on millions of materials) to "scout" the potential energy surface.
-   **Benefit**: This allows the generator to produce millions of raw candidates.
-   **Process**: The surrogate filters out unphysical structures (clashes) and identifies a geometrically diverse subset using Farthest Point Sampling (FPS).
-   **Result**: We reduce computational waste by orders of magnitude, ensuring every DFT calculation adds maximum information gain.

#### 2. Asynchronous Decoupled Architecture
The system explicitly separates the **Inference Engine** (MD simulations) from the **Learning Engine** (DFT/Training) via an asynchronous, priority-based task queue.
-   **Traditional Approach**: Stop MD -> Run DFT -> Train -> Resume MD. This leaves the GPU idle while waiting for DFT.
-   **MLIP-AutoPipe Approach**: MD continues running even while DFT is pending. If the potential becomes uncertain, the MD is paused *locally* for that job, but the Training Engine works in the background. Once a new potential is ready, it is "hot-swapped" or used for the next iteration. This maximizes hardware utilization on heterogenous HPC clusters (CPU for DFT, GPU for Training/MD).

#### 3. Physics-Informed Generation
Unlike naive random sampling, the generator module uses deep physical principles to create initial datasets.
-   **Alloys**: We use Special Quasirandom Structures (SQS) to mimic infinite random alloys in small periodic boxes.
-   **Elasticity**: We apply specific strain tensors to learn the Equation of State.
-   **Defects**: We programmatically insert vacancies and interstitials to ensure the potential is robust against radiation damage scenarios.

#### 4. The "DFT Factory" (Robust Error Recovery)
One of the biggest pain points in automation is the fragility of DFT codes. Quantum Espresso often fails to converge for metallic or magnetic systems. The "DFT Factory" module implements a **"Ladder of Robustness"**, a self-diagnosing recovery logic.
-   **Level 1**: Standard mixing.
-   **Level 2**: If convergence fails, reduce mixing beta.
-   **Level 3**: Switch diagonalization algorithm (Davidson to Conjugate Gradient).
-   **Level 4**: Increase electronic temperature (Smearing).
This logic rescues >99% of non-converging calculations without human aid, preventing pipeline stalls.

## 2. System Design Objectives

### Goals
The primary goal is to achieve fully autonomous operation.
1.  **Autonomy**: The system must operate end-to-end without manual file editing, script tweaking, or interactive shell access. Once the `mlip-auto run` command is issued, the next human interaction should be viewing the final results.
2.  **Robustness**: It must be able to handle the inherent instability of scientific software. Specifically, it must handle 99% of DFT convergence failures automatically through intelligent parameter adjustment and discard the remaining 1% safely without crashing the entire pipeline.
3.  **Efficiency**: The system must minimize the number of expensive DFT calculations. By using active learning (uncertainty quantification) and surrogate models, it should only calculate structures that add significant information to the model, rather than brute-force sampling.
4.  **Scalability**: The architecture must support distributed execution of hundreds to thousands of concurrent DFT and MD jobs, seamlessly scaling from a local 64-core node to a multi-node Slurm cluster.
5.  **Reproducibility**: All simulation parameters, data provenance (which structure came from which generation), and model versions must be tracked in a structured, queryable database (ASE-db), ensuring that every data point can be traced back to its origin.

### Constraints
1.  **External Dependencies**: The system must interface with specific, industry-standard legacy binaries: Quantum Espresso (DFT), LAMMPS (MD), and Pacemaker (Training). The system cannot modify these binaries and must communicate via file I/O and CLI arguments.
2.  **Resource Limits**: The system must respect wall-time limits and memory constraints of the underlying hardware (e.g., batch systems like Slurm). It needs mechanisms to checkpoint its state and resume gracefully if a job is killed by the scheduler.
3.  **Data Volume**: MD simulations generate terabytes of trajectory data. The system must not store raw trajectories but instead filter and store only relevant snapshots (Active Learning), keeping the database size manageable.
4.  **Language**: The core logic must be implemented in Python 3.11+, using strict type hinting (Pydantic) to ensure code quality and prevent runtime type errors in a complex, loosely typed scientific stack.

### Success Criteria
-   **End-to-End Run**: Successfully train a potential for a binary alloy (e.g., FeNi) from scratch (starting with zero data) and refine it through at least 3 generations of active learning.
-   **Simulation Stability**: Run a "Melt-Quench" simulation where the potential is updated on-the-fly, preventing the simulation from "exploding" due to unphysical extrapolations.
-   **Automation Rate**: DFT calculations must achieve >95% success rate without human intervention.
-   **Accuracy**: The final potential predicts key material properties (e.g., elastic constants, lattice parameters) within 10% of DFT reference values.

## 3. System Architecture

The system follows a modular, event-driven architecture. The central `WorkflowManager` acts as the conductor, coordinating the flow of data between specialized independent modules.

```mermaid
graph TD
    User[User Config (YAML)] --> WM[WorkflowManager]

    subgraph "Core Infrastructure"
        WM --> DB[(DatabaseManager)]
        WM --> TQ[TaskQueue (Dask)]
        WM --> Dash[Dashboard (HTML/Plotly)]
    end

    subgraph "Module A: Generator"
        WM --> Gen[StructureGenerator]
        Gen --> SQS[SQS Strategy]
        Gen --> NMS[Normal Mode Strategy]
        Gen --> Defect[Defect Strategy]
    end

    subgraph "Module B: Surrogate"
        Gen --> Sur[SurrogateExplorer]
        Sur --> MACE[MACE Model]
        Sur --> FPS[Farthest Point Sampling]
    end

    subgraph "Module C: DFT Factory"
        Sur --Candidates--> DB
        DB --Pending--> TQ
        TQ --> QER[QERunner]
        QER --> QE_Bin[Quantum Espresso]
        QER --Recovery--> QER
        QER --Results--> DB
    end

    subgraph "Module D: Training"
        DB --Labeled Data--> PM[PacemakerWrapper]
        PM --> Pace_Bin[Pacemaker]
        Pace_Bin --> Pot[Potential Artifact (.yace)]
    end

    subgraph "Module E: Inference & Active Learning"
        Pot --> Inf[InferenceRunner]
        Inf --> LAMMPS[LAMMPS MD]
        LAMMPS --Uncertainty--> UQ[UncertaintyChecker]
        UQ --High Error--> Ext[EmbeddingExtractor]
        Ext --New Candidates--> DB
        UQ --Low Error--> Prop[PropertyAnalysis]
    end
```

### Components
-   **WorkflowManager**: The brain of the operation. It implements the finite state machine that governs the pipeline lifecycle (Generation -> Surrogate Screening -> DFT Execution -> Training -> Inference -> Extraction). It monitors the database state and decides which transition to trigger next.
-   **DatabaseManager**: A unified interface for data persistence. It wraps `ase.db` to provide a SQL-like interface for storing atomic structures, energies, forces, virial stresses, and arbitrary metadata. It abstracts away the file-system level details of SQLite.
-   **TaskQueue**: An abstraction layer for parallel execution. Currently implemented using Dask Distributed, it allows the `WorkflowManager` to submit heavy compute tasks (like `QERunner.run` or `PacemakerWrapper.train`) to a pool of workers, which can be local threads or distributed SLURM jobs.
-   **Module A (Generator)**: Responsible for the "Cold Start". It uses algorithms like SQS and Rattle/Strain to produce diverse initial structures without needing a prior potential.
-   **Module B (Surrogate)**: Acts as a filter. It uses a fast, pre-trained model (MACE) to estimate the quality of generated structures, discarding those that are unphysical (e.g., overlapping atoms) and selecting the most geometrically diverse ones via Farthest Point Sampling.
-   **Module C (DFT Factory)**: A robust wrapper around Quantum Espresso. It handles input file generation, execution monitoring, and most importantly, automatic error recovery for convergence failures.
-   **Module D (Training)**: Manages the interface with the Pacemaker code, handling dataset conversion (from ASE-db to `.pcfg`), configuration generation, and training job execution.
-   **Module E (Inference)**: Runs MD simulations using LAMMPS with the trained potential. It includes "watchdogs" that monitor the extrapolation grade (uncertainty) and trigger the extraction of new training candidates when the model enters unknown territory.

## 4. Design Architecture

The codebase is structured as a Python package `mlip_autopipec`, designed with strict separation of concerns and high modularity to facilitate testing and maintenance.

### File Structure
```
mlip_autopipec/
├── app.py                  # CLI Entrypoint (Typer application)
├── config/                 # Pydantic Schemas & Configuration Logic
│   ├── models.py           # Aggregated Config Models (Top-level)
│   └── schemas/            # Granular Schemas
│       ├── common.py       # Shared types (TargetSystem, Resources)
│       ├── dft.py          # DFT specific parameters
│       └── ...             # Other module configs
├── core/                   # Core Utilities & Infrastructure
│   ├── database.py         # DatabaseManager (ASE-DB wrapper)
│   ├── logging.py          # Centralized Logger Setup
│   └── services.py         # Common Services (Config Validation)
├── data_models/            # Domain Data Objects (Runtime)
│   ├── training_data.py    # TrainingBatch, TrainingData
│   └── inference_models.py # InferenceResult, ExtractedStructure
├── dft/                    # Module C: DFT Factory
│   ├── runner.py           # QERunner (Process execution)
│   ├── inputs.py           # Input File Generator
│   └── parsers.py          # Output Parser & Validator
├── generator/              # Module A: Generator
│   ├── builder.py          # StructureBuilder (Facade)
│   └── defects.py          # DefectGenerator implementation
├── inference/              # Module E: Inference
│   ├── runner.py           # LammpsRunner
│   ├── embedding.py        # EmbeddingExtractor (Cluster cutout)
│   └── uncertainty.py      # UncertaintyChecker
├── orchestration/          # System Logic & Control Flow
│   ├── workflow.py         # WorkflowManager (State Machine)
│   └── task_queue.py       # TaskQueue (Dask wrapper)
├── surrogate/              # Module B: Surrogate
│   ├── pipeline.py         # SurrogatePipeline
│   └── sampling.py         # Farthest Point Sampling Logic
└── training/               # Module D: Training
    └── pacemaker.py        # PacemakerWrapper
```

### Key Data Models
The system relies on Pydantic models to ensure data integrity throughout the pipeline.
-   **`InferenceConfig`**: Defines all parameters required for a simulation, including temperature ranges, timesteps, ensemble types (NVT/NPT), and uncertainty thresholds.
-   **`DFTConfig`**: Encapsulates all Quantum Espresso parameters, such as wavefunction cutoffs (`ecutwfc`), K-point densities (`kspacing`), pseudopotential mappings, and SCF convergence criteria.
-   **`WorkflowState`**: A persistent object that tracks the global state of the pipeline, including the current generation index, the active phase (e.g., "TRAINING"), and the IDs of pending tasks. This enables the system to stop and resume without losing progress.
-   **`CandidateData`**: Represents an atomic structure that has been generated or extracted but not yet calculated. It contains atomic positions, cell vectors, species, and metadata (provenance).
-   **`DFTResult`**: Represents the outcome of a successful DFT calculation. It strictly enforces the presence of Total Energy (float), Forces (Nx3 array), and Stress Tensor (3x3 array), which are mandatory for training.

## 5. Implementation Plan

The project is divided into 6 sequential cycles. Each cycle builds upon the previous one, delivering a functional increment of the system.

### CYCLE01: Foundation & Configuration
-   **Objective**: Establish the project skeleton, configuration management, and persistence layer.
-   **Detailed Implementation Steps**:
    1.  **Project Initialization**: Create the `mlip_autopipec` directory structure. Set up `pyproject.toml` with `setuptools`, `ruff`, and `mypy` configurations. Create the top-level `__init__.py`.
    2.  **Configuration Schemas (Pydantic)**: Implement the core schemas in `config/schemas/common.py`. Define `TargetSystem` to validate element symbols and compositions. Create `dft.py`, `training.py`, and `inference.py` schemas with strict type checking (e.g., ensuring cutoffs are positive floats). Aggregate these into the root `MLIPConfig` model in `config/models.py`.
    3.  **Database Manager**: Develop `core/database.py`. This class must wrap `ase.db.connect`. Implement the `initialize()` method to create the SQLite file with appropriate indices (`status`, `generation`). Implement `add_structure()`, ensuring that metadata (provenance) is correctly serialized into key-value pairs. Implement `update_status()` for atomic state transitions.
    4.  **CLI Application**: Implement `app.py` using `Typer`. Create the `init` command that copies a template `input.yaml` to the user's directory. Create the `check-config` command that loads the YAML, passes it to `MLIPConfig`, and reports validation errors in a user-friendly format using `Rich`.
    5.  **Logging**: detailed implementation of `core/logging.py` to ensure all modules log to both console (info) and file (debug).
-   **Deliverables**: A functional CLI that can parse/validate complex YAML configurations and a Database Manager that can initialize and query an ASE-db SQLite database.

### CYCLE02: Physics-Informed Generator (Module A)
-   **Objective**: Implement the logic to create initial atomic structures.
-   **Detailed Implementation Steps**:
    1.  **Structure Builder Facade**: Create `generator/builder.py`. This class will accept `GeneratorConfig` and orchestrate the generation strategies.
    2.  **SQS Strategy**: Implement `generator/sqs.py`. Use `icet` (if available) or `ase` to generate Special Quasirandom Structures for alloys. This requires mapping the target composition to a supercell integer atom count (e.g., Fe0.5Ni0.5 -> 16 Fe, 16 Ni in a 32-atom cell).
    3.  **Transformation Logic**: Implement `generator/transformations.py`. Write `apply_strain` to perform affine transformations on the cell vectors ($v' = (I+\epsilon)v$). Implement `apply_rattle` to add Gaussian noise to positions.
    4.  **Defect Generation**: Implement `generator/defects.py`. Write algorithms to randomly remove atoms (vacancies) or insert atoms into Voronoi voids (interstitials).
    5.  **Integration**: Wire the `StructureBuilder` into the CLI via a `generate` command. Ensure generated structures are converted to `CandidateData` and saved to the DB with unique IDs.
-   **Deliverables**: A `StructureGenerator` capable of producing Special Quasirandom Structures (SQS) for alloys, applying elastic strains, and inserting point defects.

### CYCLE03: Surrogate Selection (Module B)
-   **Objective**: Implement the screening and selection pipeline.
-   **Detailed Implementation Steps**:
    1.  **Model Interface**: Define the `SurrogateModel` protocol in `surrogate/model_interface.py`. This ensures we can swap MACE for other models later.
    2.  **MACE Wrapper**: Implement `surrogate/mace_wrapper.py`. This class handles loading the `mace-mp-0` model using `mace-torch`. It must handle GPU/CPU device selection. Implement `compute_energy_forces` for batch processing of atoms.
    3.  **Filtering Logic**: Implement the high-force rejection filter. If predicted force > 50 eV/A, the structure is flagged as "REJECTED".
    4.  **FPS Sampling**: Implement `surrogate/sampling.py`. Write the Farthest Point Sampling algorithm. It calculates the pairwise distance matrix of atomic descriptors (extracted from MACE or SOAP) and iteratively selects the most distant points.
    5.  **Pipeline Orchestrator**: Create `surrogate/pipeline.py` to manage the flow: DB Fetch -> MACE Predict -> Filter -> FPS -> DB Update.
-   **Deliverables**: Integration with the MACE foundation model to evaluate candidates and an implementation of Farthest Point Sampling (FPS) to select the most diverse subset.

### CYCLE04: Automated DFT Factory (Module C)
-   **Objective**: Implement the robust execution environment for Quantum Espresso.
-   **Detailed Implementation Steps**:
    1.  **Input Writer**: Implement `dft/inputs.py`. Create functions to generate `pw.in` files. Crucially, ensure that `tprnfor=.true.` and `tstress=.true.` are always set, regardless of user config, as these are needed for training.
    2.  **Execution Runner**: Implement `dft/runner.py`. This class wraps `subprocess.Popen` to call `pw.x`. It manages MPI flags (`mpirun -np N`) and handles process timeouts.
    3.  **Output Parser**: Implement `dft/parsers.py`. Write regex-based or XML-based parsers to extract total energy, forces, and stress tensors from `pw.out`/`pw.xml`.
    4.  **Recovery Logic**: Implement `dft/recovery.py`. Define the `attempt_recovery` function. This implements the decision tree: if "convergence error" -> reduce mixing beta; if "diagonalization error" -> switch to CG solver.
    5.  **Integration**: Update the CLI to include a `calculate` command that runs pending jobs using this runner.
-   **Deliverables**: A `QERunner` that handles input generation, execution, output parsing, and most importantly, the auto-recovery logic for convergence failures.

### CYCLE05: Active Learning Trainer (Module D)
-   **Objective**: Implement the training interface for Pacemaker.
-   **Detailed Implementation Steps**:
    1.  **Data Exporter**: Implement `training/dataset.py`. Write the logic to query "completed" DFT results from the DB and write them to `train.xyz` and `test.xyz` in ExtXYZ format. Implement 90/10 train/test splitting.
    2.  **Pacemaker Configuration**: Implement `training/pacemaker.py`. Write the logic to generate the `input.yaml` for Pacemaker, mapping `TrainingConfig` values (cutoff, order) to the YAML format.
    3.  **Wrapper Execution**: Implement the `train()` method. It runs the `pacemaker` binary, capturing `stdout` to a log file.
    4.  **Metric Parsing**: Implement `training/metrics.py`. Parse the `log.txt` to extract RMSE values for energy and forces. Raise an error if training diverges (NaN RMSE).
    5.  **Artifact Management**: Ensure the resulting `.yace` file is correctly renamed and stored in the `potentials/` repository.
-   **Deliverables**: A `PacemakerWrapper` that converts database records into training datasets (`.pcfg`), configures the fitting process, and produces the `.yace` potential file.

### CYCLE06: Orchestration & Inference (Module E)
-   **Objective**: Implement the Active Learning loop and final orchestration.
-   **Detailed Implementation Steps**:
    1.  **MD Runner**: Implement `inference/runner.py` for LAMMPS. Write logic to generate `in.lammps` using the `pace` pair style. Include `fix halt` command to stop simulation on high uncertainty.
    2.  **Embedding Extractor**: Implement `inference/embedding.py`. Write the geometry algorithms to cut out a cluster around a high-uncertainty atom. Handle periodic boundary conditions. Generate the `force_mask` array.
    3.  **Workflow Manager**: Implement `orchestration/workflow.py`. This is the state machine. Implement `run_loop()` which checks DB counts and transitions states (GENERATION -> SELECTION -> DFT -> TRAINING -> INFERENCE).
    4.  **Task Queue**: Implement `orchestration/task_queue.py` using `dask.distributed`. Ensure tasks are submitted non-blocking.
    5.  **Dashboard**: Create a simple HTML reporter in `orchestration/dashboard.py` to visualize the pipeline status.
-   **Deliverables**: A `LammpsRunner` for MD simulations, `UncertaintyChecker` for monitoring reliability, `EmbeddingExtractor` for cutting out new candidates, and the central `WorkflowManager`.

## 6. Test Strategy

The testing strategy is built on the "Testing Pyramid" concept, emphasizing a broad base of unit tests, a substantial layer of integration tests, and targeted end-to-end tests.

### Unit Testing (Mock-Heavy)
Each module will have a dedicated test suite in the `tests/` directory.
-   **Config**: Verify validation rules strictly (e.g., ensuring temperatures are positive, cutoffs are within reasonable ranges).
-   **Generator**: Verify that SQS generation respects stoichiometry and that strain transformations mathematically correct the cell vectors.
-   **DFT**: This is critical. We will mock `subprocess.run` to simulate various Quantum Espresso exit statuses (Success, Convergence Error, Segfault). We will verify that the parser correctly extracts values from provided sample output files and that the recovery logic triggers the correct parameter adjustments.
-   **Orchestration**: We will use a "Grand Mock" strategy where all external runners (DFT, MD, Training) are replaced by mock objects. This allows us to test the state machine logic of the `WorkflowManager` (e.g., transitions from DFT -> TRAINING) without actually running heavy computations.

### Integration Testing (Component Interaction)
-   **Database**: Verify saving and retrieving complex objects. Ensure that floating-point arrays (forces) are stored with sufficient precision and that metadata queries work as expected.
-   **Pipeline Segments**: Run "mini-loops". For example, generate a structure -> save to DB -> read from DB -> write a DFT input file. This ensures that the data contracts between modules are respected.

### End-to-End (UAT)
-   **Toy System**: Run the full pipeline on a computationally cheap "Toy System" (e.g., pure Aluminum with a very small simulation box and low cutoffs).
-   **Verification**: The test passes if the system successfully completes at least one full cycle (Generate -> DFT -> Train -> MD -> Extract) automatically.
-   **Metrics**: We will check if the final potential works (i.e., MD does not crash) and if the database is populated with the expected number of calculations.
