# System Architecture

## 1. Summary

The **PyAceMaker** project aims to democratise the construction and operation of state-of-the-art Machine Learning Interatomic Potentials (MLIP), specifically leveraging the "Pacemaker" (Atomic Cluster Expansion - ACE) engine. In the modern landscape of computational materials science, bridging the gap between the high accuracy of Density Functional Theory (DFT) and the large-scale capabilities of Molecular Dynamics (MD) is a critical challenge. MLIPs offer a solution, but their creation typically requires deep expertise in both data science and computational physics, creating a high barrier to entry for many experimentalists and industrial researchers.

Current workflows often involve manual, repetitive cycles of structure generation, DFT calculation, training, and validation. These manual processes suffer from several structural issues:
1.  **Sampling Bias and Extrapolation Risk:** Standard equilibrium MD simulations often fail to capture rare events or high-energy configurations (e.g., phase transitions, chemical reactions). When an MLIP encounters these "unknown" regions during deployment, it may predict unphysical forces, leading to simulation crashes.
2.  **Inefficient Data Usage:** Generating vast amounts of correlated DFT data is computationally expensive and yields diminishing returns for potential accuracy.
3.  **High Maintenance Costs:** Fixing a "broken" potential during a simulation requires a cumbersome manual feedback loop of diagnosis, retraining, and restarting.

PyAceMaker addresses these challenges by providing a **Zero-Config Workflow**. It is an automated system that orchestrates the entire lifecycle of an ACE potential—from initial structure generation to active learning loops and final validation—controlled by a single configuration file. The system features an **Orchestrator** that manages loosely coupled modules for structure generation, DFT calculations (Oracle), training, and dynamics simulations.

Key innovations include an **Adaptive Exploration Policy** that dynamically determines sampling strategies based on material properties (e.g., switching between MD and MC based on band gap or bulk modulus), and a **Self-Healing Oracle** that automatically corrects DFT convergence errors. Furthermore, the system implements a **Hybrid Potential** strategy, overlaying a physics-based baseline (LJ/ZBL) onto the MLIP to ensure stability in the core-repulsion region, preventing unphysical atomic overlaps.

The system is designed to be highly scalable, capable of transitioning from local active learning on a workstation to large-scale simulations on HPC clusters. By automating the "Active Learning Cycle"—Exploration, Detection of uncertainty, Selection of critical structures, Calculation, Refinement, and Deployment—PyAceMaker aims to reduce the computational cost of DFT by over 90% while achieving high accuracy (Energy RMSE < 1 meV/atom).

## 2. System Design Objectives

The design of PyAceMaker is guided by the following core objectives and constraints, ensuring it meets the needs of both novice users and power users in the materials science domain.

### Goals
1.  **Zero-Config Workflow (Automation):** The primary goal is to minimise human intervention. A user should be able to define a material system and desired properties in a single `config.yaml` file, and the system should handle the rest—from initial seed generation to a fully validated production potential.
2.  **Data Efficiency (Active Learning):** The system must maximise the information gain per DFT calculation. By using uncertainty quantification (extrapolation grade $\gamma$) and D-optimality criteria (via `pace_activeset`), the system should only perform expensive quantum mechanical calculations on structures that significantly improve the potential's accuracy.
3.  **Physical Robustness (Stability):** The resulting potentials must be robust. They should not fail catastrophically when encountering high-energy collisions. This is achieved by enforcing a "Physics-Informed" baseline (ZBL/LJ) for short-range interactions, ensuring that atoms always experience repulsion at close distances, regardless of the ML model's predictions in that unsampled region.
4.  **Scalability and Modularity:** The architecture must support distinct components (Orchestrator, Generator, Oracle, Trainer, Dynamics) that can interact via defined interfaces. This allows for individual components to be upgraded or replaced (e.g., swapping QE for VASP, or LAMMPS for another MD engine) without refactoring the entire system.
5.  **Scientific Validity:** The validation module must go beyond simple RMSE metrics. It must verify physical properties such as phonon stability, elastic constants, and equations of state (EOS) to ensure the potential captures the underlying physics, not just the training data points.

### Constraints
1.  **Computational Resources:** DFT calculations are the bottleneck. The system must respect limits on concurrent calculations and handle job submissions efficiently (supporting local execution and potential scheduler integration).
2.  **External Dependencies:** The system relies on external binaries (LAMMPS, Quantum Espresso, Pacemaker, EON). These must be managed via standard environments (Docker/Singularity/Conda) to ensure reproducibility.
3.  **Error Handling:** In scientific computing, convergence failures are common. The system must be resilient, implementing "Self-Correction" logic for DFT (e.g., adjusting mixing beta) and MD (e.g., detecting and recovering from halts).

### Success Criteria
*   **Accuracy:** Energy RMSE < 1 meV/atom, Force RMSE < 0.05 eV/Å on validation sets.
*   **Efficiency:** Achieving target accuracy with < 10% of the DFT calculations required by random sampling.
*   **Reliability:** Zero "segmentation faults" during MD due to unphysical atomic overlaps (thanks to Hybrid Potentials).
*   **Usability:** A tutorial scenario (e.g., Fe/Pt on MgO) can be completed by a new user with no Python coding required.

## 3. System Architecture

The system follows a modular "Orchestrator-Worker" pattern. The **Orchestrator** is the central brain, managing the state of the active learning loop. It delegates tasks to specialised modules: **Structure Generator**, **Oracle** (DFT), **Trainer**, **Dynamics Engine**, and **Validator**.

### Components

1.  **Orchestrator:**
    *   **Role:** The state machine managing the workflow. It loads configuration, initialises modules, and executes the "Loop" (Exploration -> Detection -> Selection -> Refinement).
    *   **Responsibility:** Data management (moving files), job scheduling, and decision making (when to stop training).

2.  **Structure Generator:**
    *   **Role:** The "Explorer". It proposes new atomic configurations.
    *   **Logic:** Uses an **Adaptive Exploration Policy** to decide *how* to sample (MD vs MC, Temperature ramping, Defect injection) based on the material's "DNA" (e.g., band gap, bulk modulus).

3.  **Dynamics Engine (MD/kMC):**
    *   **Role:** The "Prover". It runs simulations using the current potential.
    *   **Logic:** Runs LAMMPS (MD) or EON (kMC). Crucially, it monitors the **Extrapolation Grade ($\gamma$)** in real-time (`fix halt`). If $\gamma$ exceeds a threshold, it halts and reports the "uncertain" structure to the Orchestrator.

4.  **Oracle (DFT):**
    *   **Role:** The "Teacher". It provides ground-truth labels.
    *   **Logic:** Runs Quantum Espresso/VASP. It features a **Self-Healing** mechanism to automatically retry failed calculations with more robust parameters (e.g., increased smearing). It also handles **Periodic Embedding** to cut out small, periodic clusters from large MD snapshots for efficient DFT.

5.  **Trainer (Pacemaker):**
    *   **Role:** The "Learner". It fits the ACE potential.
    *   **Logic:** Wraps the Pacemaker suite. It handles **Active Set Selection** (D-Optimality) to select the most informative structures from the candidates provided by the Dynamics Engine. It enforces the **Delta Learning** scheme (ACE + Physics Baseline).

6.  **Validator:**
    *   **Role:** The "Gatekeeper". It assures quality.
    *   **Logic:** Runs physical tests (Phonons, Elasticity, EOS). Only potentials that pass these physics checks are promoted to "Production".

### Data Flow Diagram

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]

    subgraph "Cycle 01: Core"
    Orch
    end

    subgraph "Active Learning Loop"
        Orch -->|1. Request Structures| Gen[Structure Generator]
        Orch -->|2. Run Simulation| Dyn[Dynamics Engine\n(LAMMPS / EON)]

        Dyn -->|3. Halt on High Uncertainty| Orch

        Orch -->|4. Select Candidates| Trainer[Trainer\n(Pacemaker)]
        Trainer -->|5. Filter (Active Set)| Oracle[Oracle\n(DFT: QE/VASP)]

        Oracle -->|6. Ground Truth (Forces/Energy)| Orch
        Orch -->|7. Update Dataset| Trainer
        Trainer -->|8. Train New Potential| Orch
    end

    subgraph "Validation & Output"
        Orch -->|9. Validate| Val[Validator]
        Val -->|Pass| Prod[Production Potential]
        Val -->|Fail| Gen
    end

    Gen -.->|Adaptive Policy| Dyn
    Oracle -.->|Self-Correction| Oracle
```

## 4. Design Architecture

The codebase is structured to enforce separation of concerns and type safety using Pydantic.

### File Structure

```ascii
src/
├── mlip_autopipec/
│   ├── core/
│   │   ├── orchestrator.py       # Main Loop Logic
│   │   ├── config_manager.py     # Global Configuration
│   │   └── exceptions.py
│   ├── domain_models/            # Pydantic Schemas
│   │   ├── config.py             # Input Config Schema
│   │   ├── structure.py          # Atom Structure Model
│   │   └── analysis.py           # Analysis Results Model
│   ├── components/
│   │   ├── generator/
│   │   │   ├── base.py
│   │   │   ├── adaptive_policy.py
│   │   │   └── builder.py
│   │   ├── oracle/
│   │   │   ├── base.py
│   │   │   ├── qe_driver.py
│   │   │   └── self_healer.py
│   │   ├── trainer/
│   │   │   ├── base.py
│   │   │   └── pacemaker_wrapper.py
│   │   ├── dynamics/
│   │   │   ├── base.py
│   │   │   ├── lammps_driver.py
│   │   │   └── eon_driver.py
│   │   └── validator/
│   │       ├── base.py
│   │       ├── phonon_calc.py
│   │       └── elastic_calc.py
│   └── utils/
│       ├── logging.py
│       └── file_ops.py
```

### Class Definitions Overview

*   **`BaseComponent`**: An abstract base class that all modules (Generator, Oracle, etc.) inherit from. It enforces a standard interface for initialisation and execution, facilitating the "strategy pattern" where different implementations (e.g., `QEOracle` vs `VASPOracle`) can be swapped seamlessly.
*   **`Orchestrator`**: The context manager. It holds instances of the components and maintains the `CycleState` (iteration number, current best potential, accumulated dataset path).
*   **`Structure` (Domain Model)**: A Pydantic wrapper around `ase.Atoms`. It ensures data integrity (e.g., checking for NaN positions) before passing data between components. It includes metadata fields for tracking provenance (e.g., "generated_by: MD_halt_step_100").
*   **`AdaptiveExplorationPolicy`**: A logic class that takes a `MaterialFeatures` object (containing band gap, etc.) and returns a `SamplingStrategy` object (defining MD temperature, MC ratio, etc.). This encapsulates the "scientific intuition" logic.

### Data Models

All configuration and data exchange will be strictly typed.
*   `Config`: The root Pydantic model representing the `config.yaml`.
*   `ValidationReport`: A structured object containing pass/fail boolean, numeric metrics (RMSE), and paths to plot images.

## 5. Implementation Plan

The project is decomposed into 8 sequential cycles.

### CYCLE 01: Core Framework
**Goal:** Establish the project skeleton, configuration management, and the central Orchestrator logic.
*   **Features:**
    *   Setup `pyproject.toml` with strict linting.
    *   Define `Domain Models` (Pydantic) for Configuration and Structures.
    *   Implement `Logging` infrastructure.
    *   Create `Abstract Base Classes` for all major components.
    *   Implement the skeletal `Orchestrator` class that can load config and instantiate dummy components.
    *   Docker/Environment setup.

### CYCLE 02: Structure Generator
**Goal:** Implement the logic to generate initial structures and the Adaptive Exploration Policy.
*   **Features:**
    *   Implement `StructureGenerator` concrete class.
    *   Develop `AdaptiveExplorationPolicy` engine that maps material properties to sampling parameters.
    *   Implement "Cold Start" strategies (using universal potentials like M3GNet if available, or random symmetry generation).
    *   Implement strategies for Defect injection (vacancies, interstitials) and Strain application.

### CYCLE 03: Oracle (DFT)
**Goal:** Automate DFT calculations with Quantum Espresso, including robust error handling.
*   **Features:**
    *   Implement `QEOracle` class (Interface to Quantum Espresso).
    *   Develop `DFTManager` for input file generation (K-points auto-grid, SSSP pseudopotentials).
    *   Implement `Self-Correction` logic: Catch crash errors and retry with safer parameters (mixing beta, smearing).
    *   Implement `Periodic Embedding`: Logic to cut out a cluster from a large structure and wrap it in a periodic box with buffer zones for accurate force calculation.

### CYCLE 04: Trainer (Pacemaker)
**Goal:** Integrate the Pacemaker engine for training potentials.
*   **Features:**
    *   Implement `PacemakerTrainer` class.
    *   Wrap `pace_train` and `pace_collect` CLI tools.
    *   Implement `Active Set Selection`: Use `pace_activeset` (MaxVol algorithm) to filter redundant structures from the training set.
    *   Data management: Handling `.pckl.gzip` datasets and merging new data.

### CYCLE 05: Dynamics (LAMMPS)
**Goal:** Enable Molecular Dynamics simulations with Hybrid Potentials.
*   **Features:**
    *   Implement `LAMMPSDynamics` class.
    *   Develop `MDInterface` to generate `in.lammps` scripts.
    *   Implement **Hybrid Potential** logic: Automatically generating `pair_style hybrid/overlay pace zbl` commands.
    *   Ensure basic MD runs (NPT/NVT) can be executed and trajectories parsed.

### CYCLE 06: OTF Loop (On-the-Fly Learning)
**Goal:** Connect the components into a closed Active Learning Loop.
*   **Features:**
    *   Implement the "Watchdog" mechanism in LAMMPS (`fix halt` based on `compute pace ... gamma_mode=1`).
    *   Implement the `Halt & Diagnose` workflow in Orchestrator.
    *   Implement `Local Candidate Generation`: Creating perturbed structures around a high-uncertainty point.
    *   Close the loop: Halt -> Extract -> Embed -> DFT -> Train -> Resume.

### CYCLE 07: Advanced Dynamics (EON/kMC)
**Goal:** Integrate Adaptive Kinetic Monte Carlo for long-timescale evolution.
*   **Features:**
    *   Implement `EONDynamics` class.
    *   Create the `pace_driver.py` bridge script that allows EON to call the ACE potential.
    *   Implement the OTF check within the EON driver (halting kMC if uncertainty is high).
    *   Orchestrate the handover between MD (fast dynamics) and kMC (slow dynamics).

### CYCLE 08: Validator & Reporting
**Goal:** Implement the Quality Assurance gatekeeper and reporting tools.
*   **Features:**
    *   Implement `StandardValidator` class.
    *   Integrate `Phonopy` for phonon dispersion stability checks.
    *   Implement Elastic Constant calculation (Born stability criteria).
    *   Implement Equation of State (EOS) fitting (Birch-Murnaghan).
    *   Generate `HTML Reports` summarizing validation metrics and plots.

## 6. Test Strategy

Testing is continuous and multi-layered.

### CYCLE 01: Core Framework
*   **Unit Tests:** Verify Pydantic model validation (ensure bad configs are rejected). Test Singleton patterns for Logging.
*   **Integration Tests:** Verify `Orchestrator` can load a dummy config and instantiate all Mock components without errors.

### CYCLE 02: Structure Generator
*   **Unit Tests:** Test `AdaptiveExplorationPolicy` logic (e.g., input "Metal" features -> output "High MC" strategy).
*   **Integration Tests:** Verify `StructureGenerator` produces valid `ase.Atoms` objects that respect the requested constraints (e.g., specific defect density).

### CYCLE 03: Oracle (DFT)
*   **Unit Tests:** Test input file generation strings (check correct flags for `tprnfor`, `tstress`). Test the "Periodic Embedding" math (box dimensions).
*   **Integration Tests (Mock):** Simulate a DFT crash and verify the `Self-Healing` logic retries with updated parameters.
*   **Integration Tests (Real):** Run a tiny SCF calculation (e.g., Si bulk) if QE is present.

### CYCLE 04: Trainer (Pacemaker)
*   **Unit Tests:** Verify command-line argument construction for Pacemaker tools.
*   **Integration Tests:** Run a dummy training on a pre-existing tiny dataset and verify a `.yace` file is produced. Test Active Set selection reduces dataset size.

### CYCLE 05: Dynamics (LAMMPS)
*   **Unit Tests:** Verify `in.lammps` string generation, specifically the `hybrid/overlay` syntax.
*   **Integration Tests:** Run a short MD step on a Mock system. Verify no "lost atoms" or immediate crashes.

### CYCLE 06: OTF Loop
*   **System Tests:** The "Mini-Loop".
    1.  Start MD.
    2.  Force a "Halt" (mock the gamma value).
    3.  Verify Orchestrator catches the halt.
    4.  Verify structure extraction and embedding.
    5.  Verify "Resume" restarts MD from the correct checkpoint.

### CYCLE 07: Advanced Dynamics (EON/kMC)
*   **Unit Tests:** Verify generation of EON `config.ini` and `reactant.con`.
*   **Integration Tests:** Run `eonclient` with the python driver. Verify the driver correctly computes energy/forces using the potential. Verify the driver raises an exit code on high uncertainty.

### CYCLE 08: Validator & Reporting
*   **Unit Tests:** Test physics calculators (e.g., Elasticity calculation from stress-strain data).
*   **System Tests:** Run the full pipeline on a toy model (e.g., LJ Argon). Verify that the Validator produces a "Green" report and the HTML is generated correctly.
