# System Architecture: PyAceMaker

## 1. Summary

PyAceMaker (Package Name: `mlip_autopipec`) is a comprehensive, automated system designed to democratise the construction and operation of Machine Learning Interatomic Potentials (MLIPs). The core engine utilises "Pacemaker" (Atomic Cluster Expansion - ACE), known for its high accuracy and computational efficiency. However, the traditional workflow for creating high-quality MLIPs is fraught with challenges: it requires deep expertise in both data science and computational physics, involves tedious manual iteration (structure generation, DFT calculation, training, validation), and often suffers from "extrapolation risks" where potentials behave unphysically in unknown regions.

PyAceMaker solves these problems by providing a "Zero-Config" workflow. It acts as an autonomous agent that orchestrates the entire lifecycle of an MLIP. From a single YAML configuration file, the system automatically generates initial structures using adaptive exploration policies, performs Density Functional Theory (DFT) calculations with self-healing error handling, trains the ACE potential with physics-informed constraints (Delta Learning), and validates the potential against rigorous physical criteria (Phonons, Elastic Constants, EOS).

The system architecture is built around a central "Orchestrator" that manages a suite of loosely coupled, container-ready components: the "Structure Generator" (Explorer), "Oracle" (DFT Calculator), "Trainer" (Pacemaker Wrapper), "Dynamics Engine" (LAMMPS/EON Interface), and "Validator" (Quality Gate). These components communicate through well-defined interfaces and shared data structures, ensuring scalability from local workstations to High-Performance Computing (HPC) environments.

A key innovation is the "Active Learning Loop," which autonomously improves the potential. The Dynamics Engine monitors the "extrapolation grade" (uncertainty) during simulations. When a simulation encounters an unknown configuration, it pauses ("Halt"), diagnoses the uncertainty, generates local candidate structures, requests accurate DFT labels from the Oracle, updates the potential, and resumes the simulation. This "Self-Driving" capability ensures that the potential becomes robust and accurate exactly where it matters most, without human intervention. By integrating Adaptive Kinetic Monte Carlo (aKMC) via EON, the system also bridges the time-scale gap, allowing the potential to learn from long-term diffusion and reaction events that are inaccessible to standard Molecular Dynamics.

## 2. System Design Objectives

The design of PyAceMaker is guided by four primary objectives, each with specific success criteria and constraints.

### 2.1. Zero-Config Workflow (Minimising Human Effort)
**Goal:** Drastically reduce the barrier to entry for non-experts. A user should be able to start a production-ready MLIP generation pipeline by providing only the chemical elements and basic physical conditions.
**Success Criteria:**
- A single `config.yaml` file drives the entire process.
- No Python coding is required for standard usage.
- The system automatically handles complex tasks like DFT convergence (mixing beta, smearing), k-point grid generation, and potential hyperparameter tuning.
- "Self-Healing" mechanisms automatically recover from common calculation errors (e.g., SCF non-convergence) without user intervention.

### 2.2. Data Efficiency (Maximising ROI)
**Goal:** Achieve high accuracy with the minimum number of expensive DFT calculations. Avoid the "Big Data" trap where redundant data consumes resources without improving the model.
**Success Criteria:**
- **Active Learning:** Use D-Optimality (MaxVol algorithm) to select only the most informative structures for training.
- **Adaptive Exploration:** Instead of random sampling, use intelligent policies (MD/MC ratio, temperature ramping, defect injection) based on the material's properties (e.g., band gap, bulk modulus).
- **Target Accuracy:** RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å using < 1/10th of the data compared to random sampling.

### 2.3. Physics-Informed Robustness (Safety First)
**Goal:** Ensure the potential never behaves unphysically, even in extrapolation regions. A "black box" ML model must not violate fundamental physical laws.
**Success Criteria:**
- **Delta Learning:** The potential learns the difference between DFT and a physical baseline (LJ or ZBL). This guarantees correct core repulsion and prevents "fusion" events in MD.
- **Hybrid Potential:** In production, the ML potential is always overlaid with the baseline (ZBL) to enforce safety limits.
- **Uncertainty Watchdog:** The system monitors the extrapolation grade ($\gamma$) in real-time and halts simulations before they explode.

### 2.4. Scalability and Extensibility (Future-Proofing)
**Goal:** The architecture must support everything from a laptop test run to a massive HPC campaign, and extend to new physics (e.g., magnetism, charge transfer) without rewriting the core.
**Success Criteria:**
- **Modular Design:** Components (Oracle, Trainer, etc.) are swappable.
- **Containerisation:** Each component can run in an isolated environment (Docker/Singularity).
- **Time-Scale Bridging:** Seamless integration of MD (ns scale) and aKMC (s to hour scale) to cover all relevant physical phenomena.

## 3. System Architecture

The system follows a "Hub-and-Spoke" architecture, where the **Orchestrator** acts as the central hub, coordinating the activities of specialised worker modules.

### Components

1.  **Orchestrator (The Brain):**
    -   Parses `config.yaml`.
    -   Manages the state of the workflow (Exploration -> Labeling -> Training -> Validation).
    -   Handles file management and directory structures for each iteration.
    -   Decides when to transition between stages based on metrics (convergence, error rates).

2.  **Structure Generator (The Explorer):**
    -   Responsible for generating candidate atomic configurations.
    -   Implements **Adaptive Exploration Policies**:
        -   *Cold Start*: Uses pre-trained models (M3GNet) to guess initial stable structures.
        -   *MD/MC*: Runs Molecular Dynamics and Monte Carlo swaps to explore phase space.
        -   *Defect Engineering*: Intelligently inserts vacancies and interstitials.
    -   Reacts to specific requests from the Validator (e.g., "Generate more strained structures").

3.  **Oracle (The Judge):**
    -   Wraps DFT codes (Quantum Espresso, VASP) via ASE.
    -   **Self-Healing**: Automatically retries failed calculations with safer parameters.
    -   **Periodic Embedding**: Cuts out "interesting" local clusters from large MD simulations and embeds them in small periodic cells for efficient DFT calculation, handling surface effects correctly.

4.  **Trainer (The Learner):**
    -   Wraps the `pacemaker` engine.
    -   Manages the training dataset (`.pckl.gzip`).
    -   Performs **Active Set Selection** (D-Optimality) to filter redundant data.
    -   Trains the ACE potential, enforcing Delta Learning against the physical baseline.

5.  **Dynamics Engine (The Worker):**
    -   Runs production simulations using the generated potential.
    -   **LAMMPS Interface**: Handles MD simulations with `hybrid/overlay` pair styles and `fix halt` for uncertainty monitoring.
    -   **EON Interface**: Handles Adaptive Kinetic Monte Carlo (aKMC) for long-time-scale evolution.
    -   **Watchdog**: Monitors the extrapolation grade $\gamma$ and signals the Orchestrator upon detecting anomalies.

6.  **Validator (The Gatekeeper):**
    -   Performs rigorous physical checks on the trained potential.
    -   **Phonon Stability**: Checks for imaginary frequencies.
    -   **Elastic Constants**: Verifies Born stability criteria.
    -   **EOS**: Checks the Equation of State for physical curvature.
    -   Generates the final quality report.

### Diagram

```mermaid
graph TD
    User[User / Config.yaml] --> Orch[Orchestrator]

    subgraph "Core Loop (Active Learning)"
        Orch -->|1. Request Structures| Gen[Structure Generator]
        Gen -->|Candidates| Orch

        Orch -->|2. Compute Properties| Oracle[Oracle (DFT)]
        Oracle -->|Forces/Energies| Orch

        Orch -->|3. Train Model| Trainer[Trainer (Pacemaker)]
        Trainer -->|Potential.yace| Orch

        Orch -->|4. Run Simulation| Dyn[Dynamics Engine]
        Dyn -->|Halt Signal / Trajectory| Orch
    end

    subgraph "Quality Gate"
        Orch -->|5. Verify| Val[Validator]
        Val -->|Pass/Fail| Orch
    end

    Dyn -.->|Uncertainty Detected| Gen
    Trainer -.->|Active Set Selection| Oracle
```

## 4. Design Architecture

The system is implemented in Python, adhering to strict software engineering principles. We use **Pydantic** for robust data validation and **Interface-based programming** to ensure modularity.

### File Structure

```ascii
src/mlip_autopipec/
├── components/
│   ├── generator/       # Structure Generator logic
│   ├── oracle/          # DFT/ASE wrappers
│   ├── trainer/         # Pacemaker wrappers
│   ├── dynamics/        # LAMMPS/EON interfaces
│   └── validator/       # Physical validation logic
├── core/
│   ├── orchestrator.py  # Main workflow logic
│   ├── config.py        # Configuration loading
│   └── state.py         # Workflow state management
├── domain_models/       # Pydantic models (Data Transfer Objects)
│   ├── config.py
│   ├── structures.py
│   └── results.py
├── utils/
│   ├── ase_adapters.py
│   └── logging.py
└── main.py              # CLI Entry point
```

### Data Models (Pydantic)

The system relies on strongly typed data models to prevent runtime errors and ensure data consistency across modules.

1.  **Configuration Models (`config.py`)**:
    -   Define the schema for `config.yaml`.
    -   Validate inputs (e.g., positive temperatures, valid element symbols).
    -   Example: `OrchestratorConfig`, `DFTConfig`, `TrainerConfig`.

2.  **Structure Models (`structures.py`)**:
    -   Standardise how atomic structures are passed between ASE, Pacemaker, and internal logic.
    -   Include metadata: `provenance` (how it was generated), `tags` (physical properties), and `status` (computed/failed).

3.  **Result Models (`results.py`)**:
    -   Encapsulate the output of calculations and validation steps.
    -   Example: `DFTResult` (energy, forces, stress, convergence_flag), `ValidationReport` (rmse, phonon_stable: bool).

### Key Classes

-   `BaseComponent`: Abstract base class for all worker modules, enforcing a standard `configure()` and `execute()` interface.
-   `Orchestrator`: The singleton controller that holds references to all components and the central state machine.
-   `AdaptivePolicy`: A logic class within the Generator that determines the next exploration strategy based on the current state (e.g., "Uncertainty is high -> Lower temperature").
-   `DFTManager`: Handles the complexity of queuing and monitoring DFT jobs, abstracting away the specifics of the underlying code (QE/VASP).

## 5. Implementation Plan

The project is divided into 8 distinct cycles, each delivering a specific slice of functionality.

### CYCLE 01: Core Framework & Configuration
**Goal:** Establish the project skeleton, configuration management, and logging infrastructure.
**Features:**
-   Project directory structure creation.
-   Pydantic models for the global configuration (`config.yaml`).
-   `Orchestrator` class skeleton with state management.
-   Logging system setup.
-   CLI entry point (`typer` based).

### CYCLE 02: Structure Generator (The Explorer)
**Goal:** Implement the logic for generating atomic structures.
**Features:**
-   `StructureGenerator` component implementation.
-   Integration with `ASE` for basic structure manipulation.
-   **Adaptive Exploration Policy**: Logic to switch between Random, MD, and Defect generation.
-   **M3GNet Integration**: "Cold Start" capability using pre-trained GNNs to guess initial structures.

### CYCLE 03: Oracle (The Judge)
**Goal:** Implement the interface to DFT codes (Quantum Espresso via ASE).
**Features:**
-   `Oracle` component implementation.
-   **DFT Manager**: Automated input generation (k-points, pseudopotentials).
-   **Self-Healing**: Error handling loop (adjust mixing beta/smearing upon failure).
-   **Periodic Embedding**: Logic to cut clusters and embed them in periodic cells.
-   Mock Oracle for CI/testing without heavy DFT.

### CYCLE 04: Trainer (The Learner)
**Goal:** Implement the interface to the Pacemaker training engine.
**Features:**
-   `Trainer` component implementation.
-   Dataset management (serialization to `.pckl.gzip`).
-   Wrapper for `pace_train`.
-   **Active Set Selection**: Wrapper for `pace_activeset` (D-Optimality).
-   **Delta Learning**: Configuration generation for `pace` to use LJ/ZBL baselines.

### CYCLE 05: Dynamics Engine (The Worker)
**Goal:** Implement the interface to LAMMPS for running simulations.
**Features:**
-   `Dynamics` component implementation.
-   Python interface to LAMMPS (via `lammps` library or subprocess).
-   **Hybrid Potential Support**: Logic to generate `pair_style hybrid/overlay` commands.
-   **Uncertainty Watchdog**: Setup of `compute pace` and `fix halt` in LAMMPS.

### CYCLE 06: The OTF Loop (Integration)
**Goal:** Connect all components into the "On-the-Fly" active learning loop.
**Features:**
-   Orchestrator logic to drive the loop: Explore -> Halt -> Label -> Train.
-   Handling of "Halt" events: Extracting the failed structure, generating local candidates, and sending to Oracle.
-   End-to-End flow verification.

### CYCLE 07: Advanced Dynamics (Time-Scale Bridging)
**Goal:** Integrate EON for long-time-scale simulations and deposition workflows.
**Features:**
-   `EONWrapper` implementation for aKMC.
-   Driver script generation for EON to call Pacemaker potentials.
-   **Deposition Module**: Logic for `fix deposit` workflows (mimicking epitaxy).
-   Seamless handover between MD (fast dynamics) and kMC (slow dynamics).

### CYCLE 08: Validator (The Quality Gate)
**Goal:** Implement physical validation checks to ensure potential quality.
**Features:**
-   `Validator` component implementation.
-   **Phonon Calculator**: Interface to `phonopy`.
-   **Elastic Calculator**: Calculation of elastic tensor and Born stability.
-   **EOS Calculator**: Birch-Murnaghan fitting.
-   HTML Report generation.

## 6. Test Strategy

Testing is critical to ensure the reliability of this complex automated system. We employ a multi-layered testing strategy.

### 6.1. Unit Testing (Pytest)
Each component is tested in isolation.
-   **Config/Models**: Verify that Pydantic models correctly validate valid inputs and reject invalid ones.
-   **Generator**: Verify that structure generation functions produce atoms objects with correct properties (cell size, number of atoms).
-   **Orchestrator**: Test state transitions (e.g., ensure it moves from "Exploration" to "Labeling" correctly).
-   **Mocking**: Heavy external dependencies (DFT, LAMMPS, Pacemaker) are mocked. For example, the `MockOracle` returns a random energy/force dictionary instead of running QE.

### 6.2. Integration Testing
Verify that components work together correctly.
-   **Generator + Oracle**: Generate a structure and pass it to the (Mock) Oracle, ensuring the data format is preserved.
-   **Trainer + Dynamics**: Train a dummy potential (using a tiny dataset) and load it into the Dynamics engine to ensure the file format and `pair_style` commands are compatible.
-   **File I/O**: Verify that the system correctly creates directories, writes logs, and saves/loads state files.

### 6.3. End-to-End (E2E) Testing / UAT
Verify the full workflow from `config.yaml` to final potential.
-   **CI/CD Pipeline**: A "Mini-Cycle" runs on every commit. It uses a very small system (e.g., minimal Si unit cell), Mock Oracle, and minimal training epochs. It must complete the full loop (Explore -> Halt -> Label -> Train) without crashing.
-   **User Acceptance Tests (UAT)**: As defined in `FINAL_UAT.md`, these are Jupyter notebooks that simulate real user scenarios (e.g., Fe/Pt deposition).
    -   **Scenario 1**: Simple bulk training.
    -   **Scenario 2**: Complex active learning with defects.
    -   **Scenario 3**: Hybrid MD/kMC simulation.

### 6.4. Physical Validation
The `Validator` component itself acts as a test suite for the *scientific* validity of the output.
-   **Regression Testing**: Ensure that the new potential is not worse than the previous generation.
-   **Sanity Checks**: "Did the system explode?" (Energy conservation in NVE MD).
