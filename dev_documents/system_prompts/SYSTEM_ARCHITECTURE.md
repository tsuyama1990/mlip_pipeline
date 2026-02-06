# System Architecture: PYACEMAKER Automated MLIP System

## 1. Summary

The **PYACEMAKER** project represents a paradigm shift in the field of computational materials science, specifically addressing the critical bottleneck of constructing high-fidelity Machine Learning Interatomic Potentials (MLIPs). As the demand for large-scale, high-accuracy atomistic simulations grows—driven by the need to understand complex phenomena such as phase transitions, chemical reactions, and material degradation—the traditional workflow for developing interatomic potentials remains labor-intensive and fraught with "expert bias."

Historically, creating a robust potential required a seasoned domain expert to manually curate datasets, tweak Density Functional Theory (DFT) parameters, and iteratively refine the fitting process. This manual approach suffers from several key deficiencies:
1.  **Human Scalability Limits:** An expert can only manage a handful of potential generation projects simultaneously.
2.  **Sampling Bias:** Humans tend to sample configurations they "expect" to be relevant, often missing rare events or high-energy states that are critical for simulation robustness.
3.  **Reproducibility Issues:** The "art" of potential fitting is often undocumented, making it difficult to reproduce or improve upon existing potentials.

**PYACEMAKER** aims to democratize this process by providing a **"Zero-Config," fully automated, closed-loop Active Learning system**. At its core, the system integrates the **Pacemaker** (Atomic Cluster Expansion) library with a sophisticated orchestration layer that autonomously navigates the vast chemical and structural space.

The system operates on a cyclic **Active Learning** principle. It begins with an initial exploration phase, using universal potentials or random sampling to generate candidate structures. These structures are evaluated for "uncertainty" or "novelty." High-uncertainty configurations are selected and passed to the **Oracle**—an automated DFT pipeline powered by Quantum Espresso or VASP. The Oracle is not merely a calculator; it is a self-correcting agent capable of diagnosing convergence failures (e.g., SCF errors) and dynamically adjusting parameters (mixing beta, smearing, k-points) to recover valuable data.

Once the "Ground Truth" data is generated, the **Trainer** module employs the Atomic Cluster Expansion (ACE) formalism to fit a potential. Crucially, the system implements a **Physics-Informed** approach by enforcing a delta-learning strategy. The MLIP learns the correction to a robust physics baseline (such as a Lennard-Jones or ZBL potential), ensuring that even in data-sparse regions (like nuclear fusion distances), the potential remains physically physically safe and does not lead to catastrophic simulation crashes.

The **Dynamics Engine**, capable of running both Molecular Dynamics (MD) via LAMMPS and Adaptive Kinetic Monte Carlo (aKMC) via EON, puts the trained potential to the test. It acts as both a validation ground and a discovery engine. By monitoring the extrapolation grade ($\gamma$) in real-time, it identifies the exact moments when the potential enters "unknown territory." These events trigger a "Halt," causing the system to extract the problematic local environment, perform a targeted DFT calculation, and retrain the model on the fly. This **Self-Healing** capability ensures that the potential becomes progressively more robust, specifically in the regions relevant to the user's simulation goals.

In summary, PYACEMAKER is not just a wrapper around existing tools; it is an **autonomous materials discovery robot**. It replaces the "Human-in-the-Loop" with a "Physics-in-the-Loop" architecture, enabling researchers to input a chemical composition and receive a production-ready, validated MLIP within days, rather than months.

## 2. System Design Objectives

The design of the PYACEMAKER system is guided by a set of rigorous objectives and constraints, ensuring that the final product is not only functional but also robust, scalable, and maintainable.

### 2.1. Democratization of MLIP Construction (Zero-Config)
The primary objective is to lower the barrier to entry.
*   **Goal:** A user with minimal knowledge of DFT or Machine Learning should be able to generate a State-of-the-Art (SOTA) potential.
*   **Constraint:** The input interface must be essentially a single configuration file (YAML). All complex hyperparameters (learning rates, basis set sizes, DFT mixing parameters) must have intelligent defaults or be auto-tuned by the system.
*   **Success Metric:** A complete "novice user" can successfully generate a potential for a binary alloy (e.g., Fe-Pt) without writing a single line of Python code.

### 2.2. Physics-Informed Robustness
We must guarantee that the generated potentials are safe for use in downstream applications.
*   **Goal:** Eliminate "simulation explosions" caused by non-physical behavior in extrapolation regions.
*   **Mechanism:** Strict enforcement of Delta Learning. The total energy is always $E_{total} = E_{baseline}(LJ/ZBL) + E_{ML}(ACE)$. The baseline provides the core repulsion.
*   **Constraint:** The system must automatically identify the appropriate baseline parameters (atomic radii, etc.) based on the chemical species involved.

### 2.3. Data Efficiency via Active Learning
DFT calculations are computationally expensive. We cannot afford to compute every possible structure.
*   **Goal:** Minimize the number of DFT calls required to reach a target accuracy (RMSE < 1 meV/atom).
*   **Mechanism:** Use **D-Optimality** (MaxVol algorithm) to select only the most information-rich structures for the training set. Discard redundant data that does not improve the model's information matrix.
*   **Success Metric:** Achieve target accuracy with < 1/10th the data compared to random sampling.

### 2.4. Autonomous Error Recovery (Self-Healing Oracle)
Automated workflows often fail due to transient numerical instabilities in DFT codes.
*   **Goal:** The system must be resilient to SCF non-convergence.
*   **Mechanism:** The Oracle module implements a "Recovery Strategy" pattern. If a calculation fails, it catches the error, analyzes the log, adjusts parameters (e.g., reduce mixing beta, increase smearing temperature), and retries.
*   **Constraint:** The system should only give up and discard a structure after multiple distinct recovery recipes have failed.

### 2.5. Scalability and Modularity
The system must support a wide range of computational resources, from a single workstation to an HPC cluster.
*   **Goal:** Decouple the logical components (Orchestrator) from the execution engines (LAMMPS, QE, Pacemaker).
*   **Mechanism:** Use abstract base classes and dependency injection. The Orchestrator talks to an `OracleInterface`, which can be implemented by `EspressoOracle`, `VaspOracle`, or even a `MockOracle` for testing.
*   **Constraint:** File-based communication (pickles, extended XYZ, JSON) is preferred over memory-based passing for large datasets to prevent OOM errors and facilitate restartability.

### 2.6. Seamless Time-Scale Bridging
MD is limited to nanoseconds; materials age over years.
*   **Goal:** Integrate MD and kMC in a unified learning loop.
*   **Mechanism:** The Dynamics Engine must abstract both LAMMPS (MD) and EON (kMC) as "Explorers." A "Halt" event in kMC (finding a high-uncertainty saddle point) is treated identically to a "Halt" in MD, triggering the same retraining workflow.

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture, with the **Orchestrator** acting as the central hub that coordinates the activities of specialized modules.

### 3.1. Components

1.  **Orchestrator (The Brain):**
    *   **Responsibility:** Manages the global state, the active learning cycle counter, and data flow between modules. It does not perform heavy computation but triggers it.
    *   **Logic:** It implements the "Check-Decide-Act" loop. It decides when to explore, when to label, when to train, and when to stop.

2.  **Structure Generator (The Explorer - Global):**
    *   **Responsibility:** Generates initial structural candidates and explores the chemical space using broad strategies (e.g., substitution, mutation, prototype enumeration).
    *   **Sub-components:**
        *   `AdaptivePolicyEngine`: Determines *how* to explore (e.g., "Heat it up" vs "Add defects") based on current model uncertainty.

3.  **Dynamics Engine (The Explorer - Local/Dynamic):**
    *   **Responsibility:** Runs simulations (MD/kMC) using the current potential to probe the phase space dynamically.
    *   **Key Feature:** **Uncertainty Watchdog**. It hooks into the simulation loop (e.g., via LAMMPS `fix halt` or Python callbacks) to interrupt the simulation the moment the extrapolation grade $\gamma$ exceeds a safety threshold.

4.  **Oracle (The Labeler):**
    *   **Responsibility:** Computes the "Ground Truth" (Energy, Forces, Stress) for candidate structures.
    *   **Key Feature:** **Periodic Embedding**. It takes a local cluster (cut from a large MD simulation) and embeds it into a periodic supercell suitable for DFT, mitigating surface effects.

5.  **Trainer (The Learner):**
    *   **Responsibility:** Fits the Interatomic Potential to the labeled dataset.
    *   **Key Feature:** **Active Set Selection**. It uses linear algebra (SVD/QR) to select the optimal subset of structures for training, keeping the dataset compact and efficient.

6.  **Validator (The Gatekeeper):**
    *   **Responsibility:** Performs independent physics checks (Phonons, Elastic Constants, EOS) to certify the potential before it is deployed.

### 3.2. Data Flow

1.  **Exploration:** Dynamics Engine runs MD. Watchdog detects High $\gamma$. Simulation Halts.
2.  **Selection:** High $\gamma$ structures are extracted. `StructureGenerator` creates local perturbations (candidates) around these failures.
3.  **Labeling:** `Oracle` performs DFT on the candidates (Self-Correcting).
4.  **Training:** `Trainer` updates the `ActiveSet` with new data and retrains the `Potential`.
5.  **Validation:** `Validator` checks the new `Potential`. If PASS, it is deployed.
6.  **Resume:** Dynamics Engine reloads the simulation state and resumes with the new `Potential`.

### 3.3. Mermaid Diagram

```mermaid
graph TD
    subgraph "Control Plane"
        Orchestrator[Orchestrator]
        Config[Global Configuration]
    end

    subgraph "Exploration Layer"
        SG[Structure Generator]
        DE[Dynamics Engine]
        MD[LAMMPS (MD)]
        KMC[EON (kMC)]
        Policy[Adaptive Policy]
    end

    subgraph "Data Generation Layer"
        Oracle[Oracle]
        DFT[DFT (QE/VASP)]
        Embed[Periodic Embedding]
    end

    subgraph "Learning Layer"
        Trainer[Trainer]
        Pace[Pacemaker]
        AS[Active Set Selector]
    end

    subgraph "Validation Layer"
        Validator[Validator]
        Phonon[Phonopy]
        Elastic[Elasticity Check]
    end

    Config --> Orchestrator
    Orchestrator --> SG
    Orchestrator --> DE
    Orchestrator --> Oracle
    Orchestrator --> Trainer
    Orchestrator --> Validator

    SG -- "Candidates" --> Oracle
    DE -- "High Uncertainty Structures" --> Oracle
    DE -- "Uses" --> Policy
    DE -- "Runs" --> MD
    DE -- "Runs" --> KMC

    Oracle -- "Labeled Data (E, F, S)" --> Trainer
    Oracle -- "Runs" --> DFT
    Oracle -- "Uses" --> Embed

    Trainer -- "Potential.yace" --> DE
    Trainer -- "Potential.yace" --> Validator
    Trainer -- "Uses" --> Pace
    Trainer -- "Uses" --> AS

    Validator -- "Validation Report" --> Orchestrator
```

## 4. Design Architecture

The system is designed with a strict **Hexagonal Architecture (Ports and Adapters)** to isolate the core domain logic from external tools (ASE, LAMMPS, Quantum Espresso).

### 4.1. File Structure

```ascii
src/
└── mlip_autopipec/
    ├── __init__.py
    ├── main.py                     # CLI Entry Point
    ├── constants.py                # Global Constants
    ├── config/
    │   ├── __init__.py
    │   └── config_model.py         # Pydantic Schemas for Config
    ├── domain_models/
    │   ├── __init__.py
    │   ├── structure.py            # Structure, Dataset Models
    │   ├── potential.py            # Potential Artifact Models
    │   └── validation.py           # Validation Result Models
    ├── interfaces/
    │   ├── __init__.py
    │   ├── explorer.py             # Abstract Base Class for Explorers
    │   ├── oracle.py               # Abstract Base Class for Oracles
    │   ├── trainer.py              # Abstract Base Class for Trainers
    │   └── dynamics.py             # Abstract Base Class for Dynamics
    ├── infrastructure/
    │   ├── __init__.py
    │   ├── explorer/
    │   │   ├── random_explorer.py
    │   │   └── adaptive_explorer.py
    │   ├── oracle/
    │   │   ├── espresso_oracle.py
    │   │   └── vasp_oracle.py
    │   ├── trainer/
    │   │   └── pacemaker_trainer.py
    │   └── dynamics/
    │       ├── lammps_adapter.py
    │       └── eon_adapter.py
    └── utils/
        ├── logging.py
        ├── file_io.py
        └── physics.py
```

### 4.2. Key Data Models (Pydantic)

We enforce type safety and schema validation using Pydantic.

*   **`GlobalConfig`**: The root configuration object. Validates the YAML input. Contains sub-configs for `OracleConfig`, `TrainerConfig`, etc.
*   **`StructureMetadata`**: Carries provenance information (source, generation method, parents) alongside the atomic coordinates (ASE Atoms).
*   **`Dataset`**: Represents a collection of labeled structures. It is not just a list of atoms; it includes metadata about the "Active Set" status and energy/force/stress units.
*   **`ValidationResult`**: A structured object containing pass/fail status, numerical metrics (RMSE, Elastic Moduli), and paths to generated plots.

### 4.3. Class Design & Interfaces

*   **`BaseOracle` (ABC)**:
    *   `compute(structures: List[Atoms]) -> List[Atoms]`: Main entry point.
    *   Implementations must handle the retrying logic internally.

*   **`BaseTrainer` (ABC)**:
    *   `train(dataset: Dataset, previous_potential: Optional[Potential]) -> Potential`: Handles the fitting process.
    *   `select_active_set(candidates: Dataset) -> Dataset`: Handles D-optimality filtering.

*   **`BaseExplorer` (ABC)**:
    *   `explore(potential: Potential, n_steps: int) -> ExplorationResult`: Runs the simulation.
    *   `ExplorationResult` contains `halted: bool`, `dump_file: Path`, `high_gamma_frames: List[int]`.

## 5. Implementation Plan

The project is decomposed into **8 Cycles**, proceeding from the core infrastructure to advanced features.

### CYCLE 01: Foundation & Orchestrator Skeleton
**Goal:** Establish the "Hello World" of the system.
**Details:**
*   Initialize the Git repository and `pyproject.toml` with strict linters.
*   Define the `GlobalConfig` Pydantic model to parse the `config.yaml`.
*   Implement the `Orchestrator` class with a dummy loop.
*   Define the Abstract Base Classes (`BaseOracle`, `BaseTrainer`, `BaseExplorer`).
*   Create `Mock` implementations of these interfaces that simply log their actions (e.g., `MockOracle` returns random energies).
*   Verify that the CLI (`main.py`) can load a config, instantiate the orchestrator, and run a "simulation" using mocks without crashing.

### CYCLE 02: The Oracle (DFT Automation)
**Goal:** Implement the real data generation engine using Quantum Espresso.
**Details:**
*   Implement `EspressoOracle` inheriting from `BaseOracle`.
*   Integrate `ase.calculators.espresso`.
*   **Key Logic:** Implement the "Self-Healing" loop. Catch `JobFailedError` or convergence errors. Implement a `RecoveryStrategy` class that suggests new parameters (mixing beta, smearing).
*   **Key Logic:** Implement `PeriodicEmbedding`. A function that takes a non-periodic cluster, boxes it in a sufficiently large supercell, and prepares it for periodic DFT.
*   **Key Logic:** Automatic pseudopotential assignment using SSSP library logic (downloading/checking pslibrary).

### CYCLE 03: The Structure Generator (Exploration Policies)
**Goal:** Create the engine that proposes new structures.
**Details:**
*   Implement `StructureGenerator` concrete class.
*   **Feature:** `InitialExploration`. Use `m3gnet` (if available) or random atomic substitution to create starting structures from a user-provided composition.
*   **Feature:** `AdaptivePolicy`. A simplified logic engine that decides what kind of structure to generate next. For this cycle, implement "Random Distortion" (Rattling) and "Volume Scaling" (EOS generation).
*   **Output:** The generator must produce `Atoms` objects with proper metadata tags.

### CYCLE 04: The Trainer (Pacemaker Integration)
**Goal:** Enable actual model training.
**Details:**
*   Implement `PacemakerTrainer` wrapping the `pace_train` and `pace_activeset` command-line tools.
*   **Feature:** Data conversion. Convert ASE atoms to Pacemaker's `.pckl.gzip` format.
*   **Feature:** Delta Learning Setup. Implement logic to write the `potential_config.yaml` with the correct `B_ref` (ZBL/LJ) settings based on species.
*   **Feature:** Active Set Selection. Implement the wrapper around `pace_activeset` to filter incoming data before training.

### CYCLE 05: Dynamics Engine Basic (MD & Inference)
**Goal:** Run MD simulations with the trained potential.
**Details:**
*   Implement `LammpsDynamics` adapter.
*   **Feature:** Input generation. Automatically write `in.lammps` files.
*   **Feature:** Hybrid Potential. Ensure `pair_style hybrid/overlay pace zbl` is correctly written.
*   **Feature:** Execution. Run LAMMPS via `subprocess` or `lammps` python interface.
*   **Feature:** Output parsing. Read `dump` files back into ASE atoms.

### CYCLE 06: Active Learning Loop (The "Halt" Mechanism)
**Goal:** Close the loop between Dynamics and Training.
**Details:**
*   **Feature:** Uncertainty Watchdog. Configure LAMMPS to compute `pace_gamma` and use `fix halt`.
*   **Feature:** Orchestrator Logic Update. Handle the `Halt` event.
    *   Detect the halt code.
    *   Read the dump file.
    *   Identify the high-$\gamma$ snapshot.
    *   Trigger the `StructureGenerator` to create candidates around this snapshot.
    *   Send to `Oracle`.
    *   Trigger `Trainer`.
    *   Resume MD.

### CYCLE 07: Advanced Exploration & Dynamics (kMC)
**Goal:** Extend the time scale with Kinetic Monte Carlo.
**Details:**
*   Implement `EonClient` adapter for `DynamicsEngine`.
*   **Feature:** EON Configuration. Auto-generate `config.ini` for EON.
*   **Feature:** Driver Script. Create the `pace_driver.py` that EON calls to get energies/forces from the `.yace` file.
*   **Feature:** Uncertainty Hook in Driver. The driver script must check $\gamma$ and kill EON if it gets too high (simulating a "Halt").
*   **Feature:** `AdaptivePolicy` upgrade. Add "Hybrid MD/MC" support.

### CYCLE 08: Validation & Production Readiness
**Goal:** Finalize the system with QA metrics.
**Details:**
*   Implement `Validator` module.
*   **Feature:** Phonon Check. Run `phonopy` to check for imaginary frequencies.
*   **Feature:** Elasticity Check. Calculate $C_{ij}$ and Bulk Modulus.
*   **Feature:** HTML Report. Aggregate all metrics into a readable report.
*   **Feature:** Full System Integration Test. Run the "Fe/Pt on MgO" scenario from start to finish.

## 6. Test Strategy

Testing is continuous and multi-layered.

### CYCLE 01 Tests
*   **Unit:** Test `GlobalConfig` validation (valid vs invalid YAML).
*   **Unit:** Test `Orchestrator` state machine transitions using Mocks.
*   **Integration:** Run the full `main.py` with `MockOracle`, `MockTrainer`, `MockExplorer`. Ensure the loop completes 3 cycles and writes log files.

### CYCLE 02 Tests
*   **Unit:** Test `PeriodicEmbedding` with a known cluster; check box size and atom positions.
*   **Unit:** Test `RecoveryStrategy` logic (e.g., input "Convergence Error", output "New Config with lower beta").
*   **Integration (Mocked DFT):** Test `EspressoOracle` flow but mock the actual `mpirun pw.x` call (intercept subprocess) to return a fake XML output. This avoids needing actual QE installed for CI.

### CYCLE 03 Tests
*   **Unit:** Test `StructureGenerator` produces valid `Atoms` objects (no overlapping atoms).
*   **Unit:** Test `AdaptivePolicy` returns different parameters for different inputs (e.g., "High Uncertainty" -> "Cautious Schedule").
*   **Visual:** Generate structures and verify with `ase.visualize`.

### CYCLE 04 Tests
*   **Unit:** Test conversion from ASE to Pacemaker format.
*   **Integration (Mocked Pace):** Mock `pace_train` execution. Check if `Trainer` correctly constructs the command line arguments and file paths.
*   **Regression:** Ensure `potential_config.yaml` is generated with the correct ZBL parameters.

### CYCLE 05 Tests
*   **Unit:** Test `in.lammps` generation. Check for `pair_style hybrid/overlay`.
*   **Integration (Mocked LAMMPS):** Mock the LAMMPS execution. Verify that the adapter can parse a standard LAMMPS dump file.

### CYCLE 06 Tests
*   **System:** "The Halt Test". Create a synthetic scenario where `MockExplorer` returns a "Halt" result immediately. Verify the Orchestrator catches it and triggers the Oracle.
*   **Logic:** Verify the data flow: Halt Structure -> Candidates -> Dataset -> New Potential.

### CYCLE 07 Tests
*   **Unit:** Test EON config generation.
*   **Integration:** Test the `pace_driver.py` script in isolation. Feed it an atom config, assert it returns energy/force in the format EON expects.
*   **Logic:** Test the "Kill Switch" in the driver when $\gamma$ is high.

### CYCLE 08 Tests
*   **Unit:** Test `Validator` metrics calculation (e.g., feed known elastic constants, check pass/fail).
*   **Full UAT:** Run the "Fe/Pt" scenario (lightweight version).
    *   Step 1: Init with 5 structures.
    *   Step 2: Train.
    *   Step 3: MD runs.
    *   Step 4: Verify output potential exists and is valid.
