# System Architecture Document

## 1. Summary

The **PYACEMAKER** project represents a significant leap forward in the democratization of computational materials science. Its primary mission is to lower the barrier to entry for constructing high-fidelity Machine Learning Interatomic Potentials (MLIPs), specifically leveraging the Atomic Cluster Expansion (ACE) framework (Pacemaker). Traditionally, creating such potentials has been the domain of experts proficient in both data science and quantum mechanics, requiring manual iteration, complex scripting, and deep intuition about sampling strategies. PYACEMAKER aims to automate this entire workflow, enabling "Zero-Config" operation where a user simply defines the material system, and the AI-driven orchestrator handles the rest.

At its core, the system utilizes a **Hub-and-Spoke** architecture. The central **Orchestrator** acts as the brain, coordinating specialized modules: the **Explorer** (Dynamics Engine/Structure Generator) which navigates the chemical and structural space; the **Oracle** (Quantum Espresso wrapper) which provides ground-truth quantum mechanical data; the **Trainer** (Pacemaker wrapper) which fits the interaction potential; and the **Validator** which rigorously tests the resulting model against physical laws.

A key innovation of this system is the **Active Learning Loop**. Instead of relying on static, pre-calculated datasets which often suffer from sampling bias, PYACEMAKER dynamically generates training data on-the-fly. It runs Molecular Dynamics (MD) simulations using the current potential, monitors the "uncertainty" (extrapolation grade) of the atomic environments, and automatically halts the simulation when it encounters a configuration it does not understand. These "confusing" structures are then selected, labeled by the Oracle (DFT), and added to the training set. This "Fail-Fast, Learn-Fast" approach ensures that the potential is specifically trained on the regions of phase space relevant to the user's simulation, achieving high accuracy with a fraction of the data required by traditional methods.

Furthermore, the system addresses the critical issue of physical robustness. Pure machine learning models can behave unpredictably in extrapolation regimes (e.g., atoms overlapping). PYACEMAKER enforces **Delta Learning**, where the ML model learns the *correction* to a robust physics-based baseline (like Lennard-Jones or ZBL). This hybrid approach guarantees that the potential remains physically sensible (i.e., repulsive at short distances) even in unexplored regions, preventing simulation crashes.

Finally, the system is designed for scale. It integrates not just MD for short-timescale dynamics, but also Adaptive Kinetic Monte Carlo (aKMC) via EON for long-timescale phenomena (diffusion, ripening). This allows the developed potentials to be used for realistic, macroscopic time-evolution studies, bridging the gap between angstroms/femtoseconds and microns/seconds.

## 2. System Design Objectives

The design of PYACEMAKER is guided by four primary objectives, each addressing a specific pain point in the current landscape of atomistic modeling.

### 2.1. Zero-Config Workflow & Automation
**Goal:** Minimize user intervention and cognitive load.
**Constraint:** The system must run from a single `config.yaml` file without requiring the user to write Python scripts or shell scripts.
**Success Criteria:** A user with no prior experience in ACE or Pacemaker should be able to generate a production-ready potential for a new alloy system (e.g., Fe-Pt) by simply specifying the elements and the desired accuracy. The system must handle all internal complexity: DFT parameter selection (k-points, smearing), error recovery (SCF convergence fixes), and hyperparameter tuning for the ML model.

### 2.2. Data Efficiency via Active Learning
**Goal:** Maximize the information content of the training set while minimizing expensive DFT calculations.
**Constraint:** DFT calculations are the bottleneck. We cannot afford to label millions of random structures.
**Success Criteria:** The system should achieve a target RMSE (Energy < 1 meV/atom, Force < 0.05 eV/Å) using 1/10th of the data compared to random sampling. It must implement "Uncertainty-Driven Sampling," where only structures with high extrapolation grades ($\gamma$) are selected for labeling. The "Active Set" optimization (D-optimality) must be used to prune redundant data.

### 2.3. Physical Robustness (Physics-Informed)
**Goal:** Prevent non-physical behavior and simulation crashes in extrapolation regimes.
**Constraint:** ML potentials are mathematical functions that can oscillate wildly outside their training data.
**Success Criteria:** The system must enforce a "Hybrid Potential" architecture. The final potential must be a sum of a physics-based baseline (ZBL/LJ) and the ML contribution (ACE). The system must demonstrate stability (no segmentation faults due to atom overlap) even at very high temperatures or under extreme compression, ensuring that the "Core Repulsion" is always dominant.

### 2.4. Scalability: From MD to kMC
**Goal:** Span multiple time and length scales.
**Constraint:** MD is limited to nanoseconds. Real materials problems (aging, corrosion) happen over seconds or hours.
**Success Criteria:** The architecture must be modular enough to swap the "Explorer" engine. It must support LAMMPS for high-speed MD (nanoseconds) and seamlessly switch to EON for aKMC (seconds/hours) using the *same* potential and the *same* active learning loop. This requires a standardized interface for "Configuration Exploration" that abstracts the underlying simulation engine.

## 3. System Architecture

The system follows a modular, service-oriented architecture (SOA) implemented in Python.

### 3.1. Component Diagram

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]

    subgraph Core System
        Orch -->|Manage| State[State Manager]
        Orch -->|Invoke| Exp[Explorer / Dynamics Engine]
        Orch -->|Invoke| Ora[Oracle / DFT]
        Orch -->|Invoke| Trn[Trainer / Pacemaker]
        Orch -->|Invoke| Val[Validator]
    end

    subgraph External Tools
        Exp -->|Run| LAMMPS[LAMMPS (MD)]
        Exp -->|Run| EON[EON (kMC)]
        Ora -->|Run| QE[Quantum Espresso]
        Trn -->|Run| Pace[Pacemaker Suite]
    end

    subgraph Data Layer
        State -->|Read/Write| DB[(Data Store / Filesystem)]
        DB -->|Load| Init[Initial Data]
        DB -->|Save| Pot[Potential.yace]
        DB -->|Log| Logs[Logs & Reports]
    end

    Exp -- High Uncertainty Structures --> Orch
    Orch -- Candidates --> Ora
    Ora -- Labeled Data --> Trn
    Trn -- New Potential --> Val
    Val -- Verified Potential --> Orch
    Orch -- Deploy --> Exp
```

### 3.2. Data Flow
1.  **Initialization**: The Orchestrator reads `config.yaml` and initializes the component interfaces.
2.  **Exploration Phase**: The Orchestrator commands the Explorer (LAMMPS/EON) to run a simulation using the current potential (or a random policy if cycle 0).
3.  **Detection**: The Explorer monitors the extrapolation grade ($\gamma$). If $\gamma >$ threshold, it halts and returns the "confusing" structure.
4.  **Selection**: The Orchestrator (or a specialized Selector) filters candidates (removing duplicates) and prepares them for the Oracle.
5.  **Labeling**: The Oracle runs DFT (Quantum Espresso) on the selected candidates to obtain ground-truth Energy, Forces, and Virals. It handles "Periodic Embedding" if the candidate is a cluster.
6.  **Training**: The Trainer updates the dataset and refines the ACE potential using Pacemaker. It enforces the physical baseline (Delta Learning).
7.  **Validation**: The Validator runs a suite of tests (Phonons, Elasticity, RMSE) on the new potential.
8.  **Deployment**: If validated, the new potential is "hot-swapped" back into the Explorer, and the simulation resumes.

## 4. Design Architecture

### 4.1. File Structure

```
mlip-pipeline/
├── pyproject.toml
├── README.md
├── config.yaml               # User configuration
├── dev_documents/            # Documentation
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── main.py           # CLI Entry Point
│       ├── config/           # Configuration Models
│       │   └── config_model.py
│       ├── domain_models/    # Pydantic Data Models
│       │   ├── structure.py  # StructureMetadata, Dataset
│       │   └── validation.py # ValidationResult
│       ├── orchestration/    # Core Logic
│       │   ├── orchestrator.py
│       │   └── state.py
│       ├── interfaces/       # Abstract Base Classes
│       │   ├── explorer.py
│       │   ├── oracle.py
│       │   ├── trainer.py
│       │   └── validator.py
│       ├── infrastructure/   # Concrete Implementations
│       │   ├── lammps/       # LAMMPS Adapter
│       │   ├── espresso/     # Quantum Espresso Adapter
│       │   ├── pacemaker/    # Pacemaker Adapter
│       │   └── eon/          # EON Adapter
│       └── utils/
│           ├── logging.py
│           └── plotting.py
└── tests/
    ├── unit/
    └── e2e/
```

### 4.2. Key Data Models (Pydantic)

*   **`StructureMetadata`**: Represents an atomic structure. Contains the `ase.Atoms` object (serialized), metadata (source, generation method), and labels (Energy, Forces).
*   **`Dataset`**: A collection of `StructureMetadata`. Managed as a persistent object (Pickle/JSON) to track the growing training set.
*   **`GlobalConfig`**: The monolithic configuration object validated at startup. Contains sub-configs for `ExplorerConfig`, `OracleConfig`, `TrainerConfig`, etc.
*   **`ValidationResult`**: Captures the outcome of a validation cycle. Includes boolean `passed`, dictionary `metrics` (RMSEs), and paths to artifacts (plots).

## 5. Implementation Plan

The development is strictly divided into **8 Cycles**.

### **CYCLE 01: Core Framework & CLI**
**Goal:** Establish the skeleton of the application, configuration system, and abstract interfaces.
**Features:**
*   Project directory structure creation.
*   `GlobalConfig` Pydantic model definition with YAML parsing.
*   `setup_logging` utility for centralized logging.
*   Abstract Base Classes (ABCs) for `Explorer`, `Oracle`, `Trainer`, `Validator`.
*   CLI entry point using `typer` (`mlip-pipeline run`).
*   **Mock implementations** of all components to allow the Orchestrator to "run" a fake loop (verify data flow without external tools).

### **CYCLE 02: Oracle Module (DFT)**
**Goal:** Implement the interface to the ground-truth provider (Quantum Espresso).
**Features:**
*   `EspressoOracle` concrete class implementation using `ase.calculators.espresso`.
*   Automatic handling of pseudopotentials (SSSP) and k-points (k-spacing logic).
*   Robust execution: Try/Except blocks to catch SCF convergence errors.
*   Simple self-healing logic (e.g., reduce mixing beta if convergence fails).
*   Input/Output management: Converting `StructureMetadata` to/from DFT input files.

### **CYCLE 03: Trainer Module (Pacemaker)**
**Goal:** Implement the interface to the MLIP training engine (Pacemaker).
**Features:**
*   `PacemakerTrainer` concrete class.
*   Wrappers for `pace_train`, `pace_collect`, and `pace_activeset` commands (via `subprocess`).
*   `Dataset` management: Merging new data into `pckl.gzip` format.
*   Delta Learning setup: Configuring the baseline potential (LJ/ZBL) in the training settings.
*   Model artifacts management: Versioning `potential.yace` files.

### **CYCLE 04: Dynamics Module (LAMMPS)**
**Goal:** Implement the interface to the MD engine (LAMMPS).
**Features:**
*   `LammpsDynamics` concrete class implementing the `Explorer` interface.
*   Dynamic generation of `in.lammps` scripts based on `GlobalConfig`.
*   Integration of `pair_style hybrid/overlay` for Delta Learning.
*   Parsing of LAMMPS `dump` files to retrieve structures.
*   Basic "Exploration" task: Running NVT/NPT MD for a fixed number of steps.

### **CYCLE 05: Validation Module**
**Goal:** Implement quality assurance checks for the generated potentials.
**Features:**
*   `Validator` concrete class.
*   Calculation of RMSE for Energy, Forces, and Virals against the Test Set.
*   Basic physical stability checks (e.g., running a short MD to ensure the system doesn't explode).
*   Generation of a summary report (JSON/HTML).
*   Integration with the Orchestrator to decide whether to "Deploy" or "Fail" a cycle.

### **CYCLE 06: Active Learning Orchestration (MVP)**
**Goal:** Connect all real components into a functioning Active Learning Loop.
**Features:**
*   Implementation of the full `Orchestrator` logic.
*   Loop Logic: Explore -> (Identify Candidates) -> Oracle -> Train -> Validate -> Deploy.
*   State persistence: Saving the current iteration state (checkpointing) to allow restart after failure.
*   Basic "Selection" logic: Randomly selecting a subset of candidates if too many are generated.
*   End-to-end execution of a simple scenario (e.g., Bulk Si) using real tools (if available) or mocks.

### **CYCLE 07: Advanced Exploration & Robustness**
**Goal:** Make the exploration "smart" and the system robust against failures.
**Features:**
*   **Adaptive Structure Generator**: Replacing random exploration with a Policy Engine (e.g., varying Temperature/Pressure based on uncertainty).
*   **Uncertainty Watchdog**: Configuring LAMMPS `fix halt` to stop simulation when extrapolation grade $\gamma$ is high.
*   **Halt & Diagnose**: Logic to extract the specific frame that caused the halt and use it as a seed for training.
*   **Recovery Strategy**: What to do if the Oracle fails or Training diverges.

### **CYCLE 08: Scale-up & Production**
**Goal:** Extend the system to long timescales and ensure production readiness.
**Features:**
*   **EON Integration**: Adding `EonExplorer` to support Adaptive Kinetic Monte Carlo (aKMC).
*   **Periodic Embedding**: Advanced logic to cut a cluster from a large MD simulation, embed it in a smaller periodic box, and run DFT (crucial for local active learning).
*   **Advanced Validation**: Phonon dispersion calculations (via Phonopy wrapper) and Elastic constant calculation.
*   **Final UAT**: Running the full "Fe/Pt on MgO" scenario.

## 6. Test Strategy

Testing is paramount for an automated system that runs expensive computations.

### 6.1. General Approach
*   **Unit Tests (`tests/unit`)**: Focus on internal logic (Config validation, Data Model integrity, String parsing). These must run fast (< 1s) and require no external binaries.
*   **Integration Tests (`tests/e2e`)**: Focus on component interactions (Orchestrator -> LammpsAdapter). These will use **Mock Objects** heavily to simulate the external binaries (LAMMPS, QE, Pacemaker) unless running in a specific "Real Mode" CI environment.

### 6.2. Cycle-Specific Testing

*   **Cycle 01**:
    *   Test `GlobalConfig` parsing with valid/invalid YAMLs.
    *   Test CLI command structure (using `typer.testing.CliRunner`).
    *   Verify Logging format.

*   **Cycle 02**:
    *   Mock `ase.calculators.espresso.Espresso`.
    *   Verify that `EspressoOracle` generates correct input files (string comparison).
    *   Verify parsing of standard QE output logs (using sample log files).

*   **Cycle 03**:
    *   Mock `subprocess.run` to simulate `pace_train`.
    *   Verify that `PacemakerTrainer` correctly assembles the command line arguments.
    *   Test dataset merging logic (using small in-memory datasets).

*   **Cycle 04**:
    *   Mock `lammps` python module or subprocess.
    *   Verify generation of `in.lammps`.
    *   Test parsing of LAMMPS dump files (using sample dump files).

*   **Cycle 05**:
    *   Test RMSE calculation logic against known vectors.
    *   Test the `Validator`'s decision logic (Pass/Fail thresholds).

*   **Cycle 06**:
    *   **Orchestration Test**: Run the full loop with Mocks for all components. Verify the sequence of calls: Explorer -> Oracle -> Trainer -> Validator.
    *   Verify state checkpointing (save/load).

*   **Cycle 07**:
    *   Test the "Watchdog" trigger logic (simulate a high-gamma return from Explorer).
    *   Test the Adaptive Policy logic (inputs -> correct parameters).

*   **Cycle 08**:
    *   Test EON configuration generation.
    *   Test Periodic Embedding logic (geometric checks: does the box contain the cluster? minimum image convention).
    *   **Final Acceptance Test**: The "Fe/Pt on MgO" tutorial notebook running in "CI Mode".
