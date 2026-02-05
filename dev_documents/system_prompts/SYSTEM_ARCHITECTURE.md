# System Architecture Document for PYACEMAKER

## 1. Summary

**PYACEMAKER** is an advanced, automated Machine Learning Interatomic Potential (MLIP) construction and operation system. It is designed to democratize the creation of State-of-the-Art (SOTA) potentials by lowering the barrier to entry for materials scientists. The core of the system is the **Pacemaker** (Atomic Cluster Expansion - ACE) tool, wrapped in an intelligent orchestration layer that automates the iterative loop of structure generation, first-principles calculation (DFT), training, and validation.

The system addresses the critical challenges in MLIP development:
1.  **Complexity**: It removes the need for manual script chaining by providing a "Zero-Config" workflow.
2.  **Data Efficiency**: It utilizes Active Learning to select only the most informative structures for DFT calculations, reducing computational costs by an order of magnitude compared to random sampling.
3.  **Robustness**: It implements physics-informed constraints (such as Delta Learning with LJ/ZBL baselines) to prevent unphysical behavior in extrapolation regions.
4.  **Scalability**: It is architected to scale from local workstations to HPC environments using containerized modules.

The ultimate goal is to allow a user to define a material system (e.g., "Fe-Pt alloy on MgO") in a single configuration file, and have the system autonomously produce a production-ready potential capable of simulating complex phenomena like hetero-epitaxial growth and phase ordering.

## 2. System Design Objectives

### 2.1. Zero-Config Workflow
*   **Goal**: Minimize user intervention.
*   **Constraint**: The user should only provide a high-level `config.yaml` defining elements and physical conditions. No Python coding should be required for standard operations.
*   **Success Criteria**: A complete novice can execute a full Active Learning cycle by running a single command `mlip-pipeline run`.

### 2.2. Data Efficiency (Active Learning)
*   **Goal**: Achieve SOTA accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with minimal DFT calculations.
*   **Constraint**: Use uncertainty metrics ($\gamma$) and D-Optimality criteria to filter "garbage" structures.
*   **Success Criteria**: Achieve convergence with <10% of the data required by random sampling methods.

### 2.3. Physics-Informed Robustness
*   **Goal**: Ensure the potential never "explodes" in MD simulations.
*   **Constraint**: Enforce Core-Repulsion using a hybrid potential scheme (ACE + LJ/ZBL).
*   **Success Criteria**: Zero segmentation faults in MD simulations, even at high temperatures or pressures, due to unphysical atomic overlaps.

### 2.4. Scalability and Extensibility
*   **Goal**: Support long-time-scale simulations and complex reaction pathways.
*   **Constraint**: Modular design allowing "plug-and-play" replacement of Oracle (QE/VASP) or Dynamics Engine (LAMMPS/EON).
*   **Success Criteria**: Seamless transition from MD-based exploration to Adaptive Kinetic Monte Carlo (aKMC) without architecture changes.

## 3. System Architecture

The system follows a **Hub-and-Spoke** architecture, where a central **Orchestrator** manages the workflow and data exchange between specialized modules.

### 3.1. Components

1.  **Orchestrator (The Brain)**:
    *   Manages the state of the Active Learning Cycle.
    *   Handles configuration loading, error recovery, and file management.
    *   Dependency Injection container for other components.

2.  **Structure Generator (The Explorer)**:
    *   **Role**: Proposes new atomic configurations.
    *   **Features**: Implements "Adaptive Exploration Policy" to switch between MD, MC, and defect engineering based on material properties (e.g., metal vs. insulator).

3.  **Oracle (The Judge)**:
    *   **Role**: Calculates ground-truth energy and forces.
    *   **Features**: Wraps DFT codes (Quantum Espresso) with "Self-Healing" capabilities to automatically fix convergence errors. Handles Periodic Embedding for cluster calculations.

4.  **Trainer (The Learner)**:
    *   **Role**: Fits the ACE potential.
    *   **Features**: Manages Delta Learning (learning the difference between DFT and a physical baseline). Uses Active Set selection to limit dataset size.

5.  **Dynamics Engine (The Executor)**:
    *   **Role**: Runs production simulations and On-the-Fly (OTF) exploration.
    *   **Features**: Integrated "Uncertainty Watchdog" (via LAMMPS `fix halt`) to detect extrapolation regions and trigger re-training. Supports aKMC via EON.

6.  **Validator (The Gatekeeper)**:
    *   **Role**: Assures quality before deployment.
    *   **Features**: Checks phonon stability, elastic constants, and basic error metrics.

### 3.2. Data Flow (Active Learning Cycle)

```mermaid
graph TD
    subgraph "Orchestration Layer"
        Orch[Orchestrator]
        Config[Global Config]
    end

    subgraph "Core Modules"
        SG[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        DE[Dynamics Engine (MD/kMC)]
        Val[Validator]
    end

    subgraph "Data Store"
        Pot[Potential (.yace)]
        Data[Dataset (.pckl)]
    end

    Config --> Orch
    Orch --> SG
    Orch --> DE

    SG -- "Candidate Structures" --> Oracle
    DE -- "High Uncertainty Structures" --> Oracle

    Oracle -- "Labeled Data (E, F, V)" --> Data
    Data --> Trainer
    Trainer -- "New Potential" --> Pot
    Pot --> Val
    Val -- "Pass/Fail" --> Orch
    Pot --> DE
```

## 4. Design Architecture

The system is built on a **Schema-First** design using Pydantic. This ensures strict validation of configuration and data exchange between modules.

### 4.1. File Structure

```ascii
src/mlip_autopipec/
├── config/
│   ├── config_model.py       # Global Pydantic Config
│   └── default.yaml          # Default Settings
├── domain_models/            # Data Exchange Schemas
│   ├── structures.py         # Atoms, Metadata
│   ├── dataset.py            # Training Data Wrappers
│   └── validation.py         # Validation Results
├── interfaces/
│   ├── core_interfaces.py    # Abstract Base Classes (Protocols)
│   └── observer.py           # Event System
├── orchestration/
│   ├── orchestrator.py       # Main Loop Logic
│   └── mocks.py              # Mock Implementations for Testing
├── structure_generation/     # Component Implementations
│   ├── generator.py
│   └── policies.py
├── oracle/
│   ├── espresso.py           # QE Wrapper
│   └── embedding.py          # Periodic Embedding Logic
├── training/
│   ├── pacemaker.py          # Pacemaker Wrapper
│   └── active_set.py         # D-Optimality Logic
├── dynamics/
│   ├── lammps_driver.py      # LAMMPS Wrapper
│   └── eon_driver.py         # EON (kMC) Wrapper
├── validation/
│   ├── metrics.py
│   └── stability.py
└── main.py                   # CLI Entry Point
```

### 4.2. Key Data Models

*   **`GlobalConfig`**: The root configuration object. Validates user input from `config.yaml`.
*   **`StructureMetadata`**: Wraps `ase.Atoms` with lineage information (source, generation method, uncertainty score).
*   **`ValidationResult`**: Contains pass/fail status and detailed metrics (RMSE, phonon stability).

## 5. Implementation Plan

The development is divided into 6 sequential cycles.

### **CYCLE 01: Core Framework & Mock Loop**
*   **Goal**: Establish the application skeleton and verify the orchestration logic using mock components.
*   **Features**:
    *   CLI entry point (`mlip-pipeline`).
    *   Configuration loading and validation (Pydantic).
    *   `Orchestrator` class implementing the main loop.
    *   `MockExplorer`, `MockOracle`, `MockTrainer` to simulate the workflow without external dependencies.
*   **Outcome**: `mlip-pipeline run` executes a full cycle (Generate -> "Calc" -> "Train") and logs progress.

### **CYCLE 02: Trainer Module Integration**
*   **Goal**: Implement real training logic using `pacemaker`.
*   **Features**:
    *   `PacemakerTrainer` class.
    *   Integration with `pace_train` and `pace_activeset` CLI tools.
    *   Implementation of "Delta Learning" configuration generation (Hybrid potential setup).
*   **Outcome**: The system can produce a valid `.yace` potential file from a provided dataset.

### **CYCLE 03: Oracle Module Integration**
*   **Goal**: Implement real DFT calculations using Quantum Espresso (via ASE).
*   **Features**:
    *   `EspressoOracle` class.
    *   Self-Healing logic (auto-adjustment of mixing beta, smearing).
    *   `PeriodicEmbedding` logic for cluster calculations.
*   **Outcome**: The system can take a structure, run a self-consistent DFT calculation, and return Energy/Forces.

### **CYCLE 04: Dynamics Engine & Active Learning Loop**
*   **Goal**: Close the loop with On-the-Fly (OTF) exploration.
*   **Features**:
    *   `LammpsDynamics` class.
    *   Integration of `fix halt` for Uncertainty Watchdog.
    *   Re-training trigger logic based on $\gamma$ threshold.
*   **Outcome**: The system runs MD, detects high uncertainty, halts, and triggers the Oracle.

### **CYCLE 05: Adaptive Structure Generation**
*   **Goal**: Replace random exploration with intelligent policies.
*   **Features**:
    *   `AdaptiveStructureGenerator`.
    *   Policy logic: Switching between MD, MC, and Defect generation based on material type (Metal vs Insulator).
    *   Initial "Cold Start" using M3GNet/Universal Potentials.
*   **Outcome**: Efficient exploration of the chemical/structural space tailored to the specific material.

### **CYCLE 06: Advanced Orchestration (kMC & Validation)**
*   **Goal**: Scale up to long time scales and ensure quality.
*   **Features**:
    *   `EonDynamics` for aKMC integration.
    *   Full `Validator` suite (Phonon stability, Elastic constants).
    *   Production-ready error handling and reporting (HTML reports).
*   **Outcome**: A robust system capable of the full "Fe/Pt on MgO" scenario.

## 6. Test Strategy

### 6.1. Unit Testing
*   **Scope**: Individual classes and functions.
*   **Tools**: `pytest`.
*   **Approach**: Mock all external calls (subprocess, filesystem). Verify logic branches (e.g., self-healing retry limits).

### 6.2. Integration Testing
*   **Scope**: Interaction between two modules (e.g., Orchestrator -> Trainer).
*   **Tools**: `pytest`.
*   **Approach**: Use temporary directories. Run actual `pacemaker` or `lammps` commands on small/toy systems if available, otherwise use high-fidelity mocks.

### 6.3. User Acceptance Testing (UAT)
*   **Scope**: End-to-End workflow.
*   **Tools**: Jupyter Notebooks, `pytest` (as driver).
*   **Approach**:
    *   **Mock Mode (CI)**: Run the full pipeline with `MockOracle` and tiny datasets. Verify the logical flow and file generation.
    *   **Real Mode**: Run the "Fe/Pt on MgO" tutorial. Verify scientific validity (e.g., potential energy < 0, stable structure).

### 6.4. CI/CD
*   **Tools**: GitHub Actions.
*   **Checks**: `ruff` (linting), `mypy` (type safety), `pytest` (unit/integration).
*   **Constraint**: All tests must pass in "Mock Mode" without requiring API keys or heavy computation.
