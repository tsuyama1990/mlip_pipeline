# System Architecture: PYACEMAKER (mlip-autopipec)

## 1. Summary

**PYACEMAKER** is a revolutionary, fully automated system designed to democratise the creation of State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) engine. Traditionally, developing high-quality MLIPs required deep expertise in both density functional theory (DFT) and machine learning, coupled with months of manual iteration. PYACEMAKER eliminates these barriers by providing a "Zero-Config" workflow that autonomously navigates the complex landscape of potential fitting.

The system orchestrates a closed-loop active learning cycle. It starts by exploring the chemical and structural space using an adaptive policy, automatically identifying regions of high uncertainty where the current potential fails or is unreliable. These "high-value" configurations are then passed to an Oracle (DFT engine) for accurate labelling. The new data is fed back into the Trainer, which refines the ACE potential. This refined potential is immediately deployed to the Dynamics Engine for further exploration.

Crucially, PYACEMAKER addresses the "extrapolation problem" inherent in ML potentials by enforcing physical baselines (Lennard-Jones/ZBL) and utilising real-time uncertainty quantification during molecular dynamics (MD) and kinetic Monte Carlo (kMC) simulations. If the simulation wanders into unknown territory, the system "halts," diagnoses the issue, learns from it, and resumes—mimicking the problem-solving process of a human expert but at the speed of software.

## 2. System Design Objectives

### 2.1. Democratisation & Usability (Zero-Config)
*   **Goal**: Enable experimentalists to generate production-grade potentials with a single configuration file (`config.yaml`), without writing Python code.
*   **Constraint**: The default settings must be robust enough to handle 80% of standard materials (metals, simple oxides) without tuning.
*   **Success Metric**: A user can install the package and start a meaningful active learning run within 10 minutes (measured by the "Time-to-First-Plot").

### 2.2. Data Efficiency (Active Learning)
*   **Goal**: Achieve DFT-level accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with minimal computational cost.
*   **Strategy**: Instead of random sampling, use "D-Optimality" to select only the most information-rich structures.
*   **Success Metric**: Reproduce the accuracy of a potential trained on 10,000 random structures using fewer than 1,000 actively selected structures.

### 2.3. Physical Robustness (Physics-Informed)
*   **Goal**: Prevent "unphysical" behaviours such as clustering atoms at a single point (nuclear fusion) or exploding simulations due to lack of repulsive forces.
*   **Strategy**: Enforce a hard physical baseline (ZBL/LJ) for short-range interactions (`pair_style hybrid/overlay`).
*   **Success Metric**: Zero segmentation faults during MD simulations, even at high temperatures (up to 2000K) or during high-energy deposition events.

### 2.4. Scalability & Extensibility
*   **Goal**: The architecture must support both small-scale cluster learning and large-scale deposition simulations.
*   **Strategy**: Modular design where components (Oracle, Trainer, Dynamics) are loosely coupled via interfaces, allowing easy replacement or scaling (e.g., swapping local DFT for cloud-based DFT).
*   **Success Metric**: Seamless transition from a single-core laptop (Mock Mode) to an HPC environment (Real Mode) via configuration flags.

## 3. System Architecture

The system follows a Hub-and-Spoke architecture, with the **Orchestrator** acting as the central brain that coordinates specialised components.

```mermaid
graph TD
    User[User / Config] -->|Initialises| Orch[Orchestrator]

    subgraph "Core Components"
        Orch -->|Configures| SG[Structure Generator]
        Orch -->|Requests Data| Oracle[Oracle (DFT)]
        Orch -->|Manages| Trainer[Trainer (Pacemaker)]
        Orch -->|Deploys Potential| Dyn[Dynamics Engine]
        Orch -->|Validates| Val[Validator]
    end

    subgraph "Data Flow (Active Learning)"
        SG -->|Candidates| Oracle
        Dyn -->|Halt: High Uncertainty| Orch
        Orch -->|Diagnosis & Selection| SG
        Oracle -->|Labelled Data| Trainer
        Trainer -->|New Potential.yace| Dyn
        Trainer -->|New Potential.yace| Val
    end

    subgraph "External Engines"
        Oracle -.->|Calls| QE[Quantum Espresso]
        Trainer -.->|Calls| Pace[Pacemaker]
        Dyn -.->|Calls| LAMMPS[LAMMPS]
        Dyn -.->|Calls| EON[EON (aKMC)]
    end
```

### Component Roles
1.  **Orchestrator**: The state machine manager. It decides "what to do next" (Exploration, Labelling, Training, or Validation) based on the current state and configuration.
2.  **Structure Generator**: Produces atomic structures. It supports "Cold Start" (using heuristics/M3GNet) and "Local Refinement" (generating candidates around a failed MD snapshot).
3.  **Oracle**: The ground truth provider. It wraps DFT codes (Quantum Espresso) with self-healing logic to handle convergence errors automatically.
4.  **Trainer**: Wraps the Pacemaker engine. It manages the training dataset, selects active sets (D-Optimality), and fits the ACE potential.
5.  **Dynamics Engine**: The "user" of the potential. It runs MD (LAMMPS) or kMC (EON). It features a "Watchdog" that halts execution if the uncertainty metric ($\gamma$) exceeds a safety threshold.
6.  **Validator**: Independent auditor. It runs physical tests (Phonon stability, Elastic constants, EOS) to ensure the potential is not just numerically accurate but physically sound.

## 4. Design Architecture

The codebase is structured to enforce separation of concerns, utilising Pydantic for robust data validation and strict typing.

### 4.1. File Structure (ASCII Tree)

```ascii
src/mlip_autopipec/
├── main.py                     # CLI Entry Point
├── config.py                   # Global Configuration Loading
├── constants.py                # Physical Constants & Default Settings
├── core/                       # Core Logic
│   ├── orchestrator.py         # Main Workflow Manager
│   ├── state_manager.py        # Persistence (Checkpointing)
│   └── logger.py               # Centralised Logging
├── domain_models/              # Pydantic Data Models (The "Contract")
│   ├── config.py               # Configuration Schemas
│   ├── datastructures.py       # Structure, Dataset, Potential Objects
│   └── enums.py                # Enums (Status, TaskTypes)
└── components/                 # Functional Modules
    ├── base.py                 # Abstract Base Classes
    ├── generators/             # Structure Generation Logic
    │   ├── random.py
    │   └── adaptive.py
    ├── oracle/                 # DFT Interfaces
    │   ├── qe.py
    │   └── mock.py
    ├── training/               # Pacemaker Wrappers
    │   ├── pacemaker.py
    │   └── activeset.py
    ├── dynamics/               # MD/kMC Engines
    │   ├── lammps.py
    │   └── eon.py
    └── validation/             # Physics Validation
        ├── phonons.py
        └── elasticity.py
```

### 4.2. Key Data Models

*   **`Structure`**: A wrapper around `ase.Atoms` that includes metadata (provenance, energy, forces, uncertainty). It is the primary currency of data exchange.
*   **`Potential`**: Represents a trained model version, including path to `.yace` file, training metrics, and validation status.
*   **`WorkflowState`**: A JSON-serializable object that tracks the current iteration, completed tasks, and path to the latest potential. This allows the system to resume after a crash.

## 5. Implementation Plan

The development is divided into 8 sequential cycles, each building upon the previous one to ensure a stable and testable progression.

### **CYCLE 01: Core Framework & Orchestration**
*   **Goal**: Establish the project skeleton, CLI, logging, and the basic Orchestrator loop using Mock components.
*   **Deliverables**: A runnable `mlip-auto run` command that cycles through mock stages (Gen -> Oracle -> Train) without doing real physics.
*   **Verification**: Logging output confirms the correct state transitions.

### **CYCLE 02: Structure Generator & Adaptive Policy**
*   **Goal**: Implement the `StructureGenerator` capable of producing initial random structures and applying adaptive policies (e.g., "High-T MD", "Strain-Scanning").
*   **Deliverables**: `AdaptiveGenerator` class that reads policy config and outputs `ase.Atoms` objects.
*   **Verification**: Generation of diverse structures (distorted lattices, vacancies) confirmed by visualisation.

### **CYCLE 03: Oracle (DFT Automation)**
*   **Goal**: Implement the `DFTOracle` with Quantum Espresso integration.
*   **Deliverables**: A robust DFT pipeline that handles input file generation, execution, and error recovery (Self-Healing). Implementation of "Periodic Embedding" for cluster calculations.
*   **Verification**: Successful SCF calculation of a test structure; automatic retry upon forced convergence failure.

### **CYCLE 04: Trainer (Pacemaker Integration)**
*   **Goal**: Integrate the Pacemaker engine for potential fitting.
*   **Deliverables**: `PacemakerTrainer` that manages datasets, runs `pace_train`, and implements Active Set selection (D-Optimality) to filter redundant data.
*   **Verification**: Training a dummy potential on a small dataset and reducing the dataset size using active set selection.

### **CYCLE 05: Dynamics Engine (LAMMPS & Uncertainty)**
*   **Goal**: Implement `LAMMPSDynamics` with the critical "Watchdog" feature.
*   **Deliverables**: MD engine that runs with `pair_style hybrid/overlay` (ACE + ZBL) and halts when extrapolation grade $\gamma$ exceeds a threshold.
*   **Verification**: Simulation halts correctly when a "strange" structure is encountered; ZBL prevents atoms from overlapping.

### **CYCLE 06: The On-the-Fly (OTF) Loop**
*   **Goal**: Close the loop. Connect Dynamics Halt -> Local Candidate Generation -> Oracle -> Trainer -> Resume.
*   **Deliverables**: Fully functional Active Learning loop where the system autonomously improves the potential based on MD failures.
*   **Verification**: An automated run that starts with a poor potential, crashes (halts), learns, and eventually stabilizes.

### **CYCLE 07: Advanced Dynamics (EON & Deposition)**
*   **Goal**: Integrate Kinetic Monte Carlo (EON) and complex MD scenarios (Deposition).
*   **Deliverables**: `EONWrapper` for long-timescale evolution. `Deposition` module for simulating film growth.
*   **Verification**: Simulation of Fe/Pt deposition on MgO, bridging MD (deposition) and kMC (ordering).

### **CYCLE 08: Validation & Reporting**
*   **Goal**: Implement the Quality Assurance gate.
*   **Deliverables**: `StandardValidator` calculating Phonons, Elastic Constants, and EOS. Generation of HTML reports.
*   **Verification**: A final "Green/Red" report generated after training, correctly identifying physical instabilities.

## 6. Test Strategy

Testing is continuous and multi-layered, ensuring no regression at each cycle.

### 6.1. Unit Testing (`pytest`)
*   **Scope**: Individual classes and functions (e.g., `Config.load()`, `Structure.to_dict()`).
*   **Strategy**: Mock all external binaries (LAMMPS, QE, Pacemaker). Test logic paths, error handling, and data validation.
*   **Metric**: >80% code coverage.

### 6.2. Integration Testing
*   **Scope**: Interaction between two components (e.g., `Orchestrator` -> `DFTOracle`).
*   **Strategy**: Use "Mock" versions of heavy engines but real file I/O. Ensure data flows correctly from one component to another (e.g., Structure -> Input File -> Output File -> Structure).

### 6.3. System/E2E Testing (The "UAT")
*   **Scope**: Full workflow execution.
*   **Strategy**:
    *   **CI Mode**: Run the full loop using `Mock` components or very small systems (LJ potential, 2 atoms) to verify the *logic* of the pipeline in < 5 minutes.
    *   **Production Mode**: Run the actual `Fe/Pt on MgO` scenario on a workstation to verify scientific validity.
*   **Artifacts**: Jupyter Notebooks in `tutorials/` serve as executable acceptance tests.
