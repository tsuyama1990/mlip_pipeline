# System Architecture: PYACEMAKER

## 1. Summary
PYACEMAKER is a cutting-edge automated system designed to democratize the creation and operation of Machine Learning Interatomic Potentials (MLIP). At its core, it leverages the Atomic Cluster Expansion (ACE) formalism via the Pacemaker engine to generate high-accuracy potentials that rival First-Principles (DFT) calculations in precision while maintaining the computational efficiency of classical molecular dynamics (MD).

The primary objective of PYACEMAKER is to bridge the gap between computational physics experts and materials scientists who may lack deep expertise in MLIP construction. Traditional MLIP workflows are often disjointed, requiring manual intervention at multiple stages: structure generation, DFT calculation, data curation, training, and validation. This manual process is error-prone, time-consuming, and often leads to "black-box" potentials that fail catastrophically in unknown regions of the phase space.

PYACEMAKER addresses these challenges by providing a "Zero-Config" workflow. Users define their material system and desired properties in a single configuration file, and the system autonomously orchestrates the entire lifecycle of the potential. This includes an "Adaptive Exploration Policy" that intelligently samples the configuration space—focusing on physically relevant structures like defects, surfaces, and high-temperature distortions—rather than relying on random sampling.

The system features a robust "Active Learning" loop. It starts with a "Cold Start" phase using universal potentials (like M3GNet) to get a rough energy landscape. Then, it iteratively refines the ACE potential by running MD simulations, detecting high-uncertainty configurations (via the extrapolation grade $\gamma$), and automatically triggering DFT calculations for those specific structures. This "Halt & Diagnose" mechanism ensures that the potential is only trained on data that actually improves its accuracy, maximizing data efficiency.

Furthermore, PYACEMAKER integrates advanced simulation capabilities directly into the workflow. It seamlessly couples MD (LAMMPS) for fast dynamics with Adaptive Kinetic Monte Carlo (aKMC via EON) to explore long-time-scale phenomena like diffusion and ripening. To ensure physical robustness, it employs a hybrid potential strategy, enforcing a physics-based baseline (LJ/ZBL) for core-repulsion to prevent non-physical atomic overlaps. Finally, a comprehensive Validator module automatically assesses the potential's quality against rigorous physical criteria—phonon stability, elastic constants, and EOS curves—before deploying it for production use.

## 2. System Design Objectives

### 2.1. Zero-Config Automation
**Goal:** Minimize human intervention.
The system must be capable of taking a simple input (e.g., "Fe-Pt alloy") and delivering a production-ready potential without requiring the user to write Python scripts or manually manage file formats.
**Success Criteria:**
- A complete training cycle (Generation -> DFT -> Train -> Validate) runs from a single `config.yaml`.
- Automatic error recovery for DFT calculations (e.g., SCF convergence failures).

### 2.2. Data Efficiency & Active Learning
**Goal:** Maximize accuracy with minimal DFT cost.
Instead of brute-force random sampling, the system must use Active Learning to select only the most informative structures. The target is to achieve DFT-level accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with 1/10th the data of traditional methods.
**Success Criteria:**
- Implementation of "Active Set Optimization" (D-optimality) to filter redundant structures.
- On-the-fly uncertainty quantification ($\gamma$) to trigger retraining only when necessary.

### 2.3. Physics-Informed Robustness
**Goal:** Prevent unphysical behavior.
Machine learning models often behave unpredictably outside their training data. PYACEMAKER must guarantee basic physical sanity, such as preventing atoms from fusing (core collapse).
**Success Criteria:**
- Strict enforcement of Hybrid Potentials (ACE + LJ/ZBL) in all MD/kMC simulations.
- Automatic validation of Phonon spectra to ensure dynamic stability (no imaginary frequencies).

### 2.4. Scalability & Modularity
**Goal:** Support diverse computing environments.
The architecture must be modular to run on local workstations or HPC clusters. Components (Oracle, Trainer, Dynamics) should be loosely coupled.
**Success Criteria:**
- Clear separation of concerns via Abstract Base Classes.
- Container-friendly design (Docker/Singularity) for reproducibility.

## 3. System Architecture

The system is designed as a modular Orchestrator-Worker pattern. The central **Orchestrator** manages the workflow state and delegates tasks to specialized modules: **Structure Generator**, **Oracle** (DFT), **Trainer** (ML), **Dynamics Engine** (MD/kMC), and **Validator**.

### 3.1. Component Diagram

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]

    subgraph "Core System"
        Orch -->|Request Structures| Gen[Structure Generator]
        Orch -->|Submit Jobs| Oracle[Oracle (DFT)]
        Orch -->|Train Model| Trainer[Trainer (Pacemaker)]
        Orch -->|Run Sim| Dynamics[Dynamics Engine]
        Orch -->|Verify| Valid[Validator]
    end

    subgraph "External Tools"
        Gen -.->|M3GNet| M3G[Universal Potential]
        Oracle -.->|QE/VASP| DFT[DFT Code]
        Trainer -.->|PACE| PACE[Pacemaker]
        Dynamics -.->|LAMMPS| LMP[LAMMPS]
        Dynamics -.->|EON| EON[EON Client]
        Valid -.->|Phonopy| Phono[Phonopy]
    end

    subgraph "Data Storage"
        DB[(Dataset .pckl)]
        Pot[(Potential .yace)]
    end

    Oracle -->|Forces/Energy| DB
    Trainer -->|Read| DB
    Trainer -->|Write| Pot
    Dynamics -->|Read| Pot
    Dynamics -->|High Uncertainty| Orch
```

### 3.2. Data Flow
1.  **Initialization**: The Orchestrator reads `config.yaml` and initializes the modules.
2.  **Exploration (Cold Start / MD)**:
    - **Cold Start**: The Structure Generator creates initial random/perturbed structures or uses M3GNet to find approximate minima.
    - **Active Learning**: The Dynamics Engine runs MD with the current potential. It monitors the extrapolation grade $\gamma$. If $\gamma > \text{threshold}$, it halts and returns the "bad" structure.
3.  **Selection**: The Orchestrator (or Trainer) selects the most representative structures from the "bad" candidates using D-optimality (MaxVol algorithm).
4.  **Labeling**: The Oracle runs DFT calculations on the selected structures to get ground-truth forces and energies. It handles error recovery automatically.
5.  **Training**: The Trainer updates the dataset and fine-tunes the ACE potential using the new data. It ensures the potential is hybrid (ACE + Baseline).
6.  **Validation**: The Validator runs physics checks (Phonons, EOS). If passed, the potential is deployed for the next cycle.

## 4. Design Architecture

The system is built on a rigorous Object-Oriented Design (OOD) with strict typing (Pydantic) to ensure robustness and maintainability.

### 4.1. File Structure
```text
src/pyacemaker/
├── core/                   # Core infrastructure
│   ├── base.py             # Abstract Base Classes
│   ├── config.py           # Pydantic Configuration Models
│   ├── logging.py          # Centralized Logging
│   └── exceptions.py       # Custom Exceptions
├── oracle/                 # DFT Automation
│   ├── manager.py          # Oracle Manager
│   ├── calculator.py       # ASE Calculator Wrappers (QE, VASP)
│   └── dataset.py          # Dataset Management (pckl.gzip)
├── trainer/                # Training Engine
│   ├── wrapper.py          # Pacemaker CLI Wrapper
│   └── active_set.py       # Active Set Selection Logic
├── generator/              # Structure Generation
│   ├── policy.py           # Adaptive Exploration Policy
│   ├── strategies.py       # Random, M3GNet, Defect strategies
│   └── mutations.py        # Atomic perturbations
├── dynamics/               # Simulation Engine
│   ├── md.py               # LAMMPS Interface & Halt Logic
│   ├── kmc.py              # EON Interface
│   └── potential.py        # Hybrid Potential Management
├── validator/              # Quality Assurance
│   ├── physics.py          # Phonon, EOS, Elasticity checks
│   └── report.py           # HTML Report Generation
└── main.py                 # CLI Entry Point
```

### 4.2. Key Design Patterns
-   **Strategy Pattern**: Used in `StructureGenerator` to switch between different exploration strategies (e.g., `RandomPerturbation`, `M3GNetRelax`, `DefectInjection`) based on the Adaptive Policy.
-   **Factory Pattern**: Used in `Oracle` to instantiate the correct DFT calculator (QE or VASP) based on configuration.
-   **Observer Pattern**: The `DynamicsEngine` monitors the simulation state. When uncertainty spikes, it notifies the Orchestrator (Subject-Observer) to trigger the active learning loop.
-   **Pydantic Models**: All configuration and data exchange objects (e.g., `StructureData`, `TrainingConfig`, `ValidationResult`) are defined as Pydantic models. This ensures runtime type validation and easy serialization/deserialization.

### 4.3. Data Models (Conceptual)
-   `StructureData`: Wraps `ase.Atoms` with metadata (provenance, energy, forces, virial, uncertainty_score).
-   `PotentialArtifact`: Represents a trained potential file (`.yace`) with its version, validation status, and training metrics.
-   `ExplorationState`: Tracks the current coverage of the chemical/configurational space to guide the Policy.

## 5. Implementation Plan

The development is divided into 6 sequential cycles.

### **Cycle 01: Core Framework & Configuration**
-   **Goal**: Establish the project skeleton and configuration management.
-   **Features**:
    -   CLI entry point (`pyace`).
    -   Pydantic-based configuration loading (`config.yaml`).
    -   Centralized logging system.
    -   Abstract Base Classes (`BaseModule`, `BaseOracle`, `BaseTrainer`).
    -   Basic `Orchestrator` shell.

### **Cycle 02: Oracle & Data Management**
-   **Goal**: Enable automatic DFT calculations and dataset handling.
-   **Features**:
    -   `DFTManager` integrating ASE with Quantum Espresso.
    -   Automatic generation of DFT inputs (k-spacing, pseudos).
    -   Self-healing DFT logic (handling SCF convergence errors).
    -   `DatasetManager` for reading/writing Pacemaker-compatible datasets (`.pckl.gzip`).

### **Cycle 03: Trainer & Potential Generation**
-   **Goal**: Integrate Pacemaker for potential training.
-   **Features**:
    -   `PacemakerWrapper` to call `pace_train`, `pace_collect`.
    -   Implementation of Delta Learning (configuring LJ/ZBL baselines).
    -   `Trainer` module to manage training jobs and versioning.
    -   Basic Active Set selection (D-optimality) interface.

### **Cycle 04: Structure Generator & Adaptive Policy**
-   **Goal**: Smart generation of candidate structures.
-   **Features**:
    -   `StructureGenerator` module.
    -   `AdaptivePolicy` engine to decide sampling strategies.
    -   Implementation of strategies: Random Perturbation, Supercell creation, Defect generation.
    -   Integration of "Cold Start" using M3GNet (optional/mock for now).

### **Cycle 05: Dynamics Engine & On-the-Fly Learning**
-   **Goal**: Close the Active Learning loop with MD.
-   **Features**:
    -   `MDInterface` for LAMMPS.
    -   Automatic `in.lammps` generation with `pair_style hybrid/overlay`.
    -   `fix halt` integration for uncertainty-based stopping.
    -   The "Halt & Diagnose" loop: Extract bad structure -> Embed -> Send to Oracle.

### **Cycle 06: Validation & kMC Integration**
-   **Goal**: Ensure quality and extend time scales.
-   **Features**:
    -   `Validator` module: Phonon dispersion, EOS curves, Elastic constants.
    -   `EONWrapper` for Adaptive kMC integration.
    -   HTML Report generation.
    -   Full System Integration (Fe/Pt on MgO scenario).

## 6. Test Strategy

We employ a "Testing Pyramid" approach, emphasizing unit tests for stability and integration tests for workflow verification.

### 6.1. Unit Testing (Pytest)
-   **Coverage**: Target > 80% code coverage.
-   **Mocking**: Heavy use of `unittest.mock` to simulate external binaries (LAMMPS, QE, Pacemaker). We will not run actual DFT or MD in unit tests.
-   **Pydantic Validation**: Tests to ensure invalid configurations are caught early.

### 6.2. Integration Testing
-   **Module Integration**: Verify that `Orchestrator` correctly passes data between `Generator` and `Oracle`, or `Trainer` and `Dynamics`.
-   **File I/O**: Verify that `.pckl` and `.yace` files are correctly read/written.

### 6.3. System/UAT Testing
-   **Mock Mode**: A special "CI Mode" configuration where external heavy computations (DFT, MD) are replaced by dummy data generators. This allows the full orchestrator loop to be tested in minutes on GitHub Actions.
-   **Real Mode**: Full execution on a local machine with actual binaries installed.
-   **Scenario**: The "Fe/Pt on MgO" scenario will be the primary UAT case, verifying the full pipeline from Cold Start to kMC ordering.

### 6.4. Continuous Integration (CI)
-   **Linting**: `ruff` and `mypy` (strict) runs on every commit.
-   **Tests**: `pytest` runs on every PR.
