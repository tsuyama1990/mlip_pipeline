# System Architecture: PYACEMAKER

## 1. Summary
PYACEMAKER is an automated system for constructing and operating Machine Learning Interatomic Potentials (MLIP) using the "Pacemaker" (Atomic Cluster Expansion) engine. The primary goal is to democratise high-accuracy atomic simulations by enabling users with minimal expertise to generate "State-of-the-Art" potentials.

The system addresses the challenges of traditional MLIP construction, such as the high barrier to entry, the risk of unphysical extrapolation, and the inefficiency of random sampling. By automating the loop of exploration, detection, selection, calculation, and refinement, PYACEMAKER ensures robust and efficient potential generation.

## 2. System Design Objectives

### 2.1 Zero-Config Workflow
The system allows users to execute a complete pipeline—from initial structure generation to training completion—using a single YAML configuration file. No custom Python scripting is required from the user.

### 2.2 Data Efficiency
By combining Active Learning with physics-based sampling, the system aims to achieve high accuracy (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with significantly fewer DFT calculations compared to random sampling.

### 2.3 Physics-Informed Robustness
The system enforces physical correctness, particularly in core repulsion, by using Delta Learning with a Lennard-Jones or ZBL baseline. This ensures that simulations do not crash due to atomic overlap, even in extrapolation regions.

### 2.4 Scalability and Extensibility
The modular architecture supports seamless scaling from local active learning to large-scale Molecular Dynamics (MD) and Kinetic Monte Carlo (kMC) simulations.

## 3. System Architecture

The system follows a modular architecture orchestrated by a central Python controller.

### 3.1 Components
1.  **Orchestrator**: The central brain that manages the workflow, state transitions, and data flow between components.
2.  **Structure Generator**: explores the chemical and structural space to propose candidate structures using an Adaptive Exploration Policy.
3.  **Oracle**: Performs Density Functional Theory (DFT) calculations (using Quantum Espresso) to label structures with energy, forces, and stress. It includes self-healing mechanisms for calculation failures.
4.  **Trainer**: Wraps the Pacemaker engine to train the ACE potential. It implements Active Set optimisation and Delta Learning.
5.  **Dynamics Engine**: Executes MD (LAMMPS) and kMC (EON) simulations. It performs On-the-Fly (OTF) uncertainty monitoring to detect when the potential enters unknown regions.
6.  **Validator**: rigorous quality assurance module that verifies the potential against physical properties (phonons, elasticity, EOS).

### 3.2 Mermaid Diagram

```mermaid
graph TD
    User[User] -->|Config (yaml)| Orch[Orchestrator]
    Orch -->|Control| Gen[Structure Generator]
    Orch -->|Control| Dyn[Dynamics Engine]
    Orch -->|Control| Oracle[Oracle (DFT)]
    Orch -->|Control| Trainer[Trainer (Pacemaker)]
    Orch -->|Control| Valid[Validator]

    Gen -->|Candidate Structures| Dyn
    Dyn -->|Exploration (MD/kMC)| OTF{Uncertainty Check}
    OTF -->|High Uncertainty| Oracle
    OTF -->|Low Uncertainty| Dyn
    Oracle -->|Labeled Data| Dataset[(Dataset)]
    Dataset --> Trainer
    Trainer -->|Potential (yace)| Dyn
    Trainer -->|Potential (yace)| Valid
    Valid -->|Report| User
```

## 4. Design Architecture

### 4.1 File Structure

```ascii
src/
├── mlip_autopipec/
│   ├── __init__.py
│   ├── main.py                 # CLI Entry Point
│   ├── core/
│   │   ├── orchestrator.py     # Main Loop
│   │   ├── dataset.py          # Data Management
│   │   └── state.py            # State Management
│   ├── components/
│   │   ├── generator/          # Structure Generator
│   │   ├── oracle/             # DFT Interface
│   │   ├── trainer/            # Pacemaker Interface
│   │   ├── dynamics/           # LAMMPS/EON Interface
│   │   └── validator/          # QA Module
│   ├── domain_models/          # Pydantic Models
│   │   ├── config.py
│   │   ├── structure.py
│   │   └── potential.py
│   ├── interfaces/             # Abstract Base Classes
│   └── utils/
```

### 4.2 Data Models (Pydantic)

The system relies on strict data validation using Pydantic.

*   **GlobalConfig**: Validates the user's YAML configuration.
*   **Structure**: Represents an atomic structure with positions, numbers, cell, PBC, and optional labels (energy, forces, stress).
*   **Potential**: Represents a trained potential file and its metadata.

## 5. Implementation Plan

The project is divided into 6 sequential implementation cycles.

### CYCLE 01: Core Framework & Mocks
*   **Goal**: Establish the project structure, CLI, configuration loading, and orchestration logic using mock components.
*   **Features**:
    *   Project skeleton and dependency management.
    *   `GlobalConfig` Pydantic model.
    *   `Orchestrator` class with a mock loop.
    *   Base interfaces for all components.
    *   Mock implementations for Generator, Oracle, Trainer, Dynamics.
    *   Logging system.

### CYCLE 02: Data Management & Structure Generator
*   **Goal**: Implement robust dataset handling and the adaptive structure generator.
*   **Features**:
    *   `Dataset` class for managing atomic structures (JSONL/Pickle).
    *   `Structure` domain model with validation.
    *   `StructureGenerator` implementation with Adaptive Exploration Policy (stub logic for now, or basic random/heuristic).
    *   Integration of `pymatgen` or `ase` for structure manipulation.

### CYCLE 03: Oracle (DFT Interface)
*   **Goal**: Implement the interface to Quantum Espresso for robust DFT calculations.
*   **Features**:
    *   `Oracle` component implementation.
    *   Automatic input file generation (pseudopotentials, k-points).
    *   Self-healing mechanism for SCF convergence failures.
    *   Periodic Embedding for cluster calculations.

### CYCLE 04: Trainer (Pacemaker Integration)
*   **Goal**: Integrate the Pacemaker engine for potential training.
*   **Features**:
    *   `Trainer` component implementation.
    *   Wrapper for `pace_train` and `pace_activeset`.
    *   Delta Learning setup (LJ/ZBL baseline).
    *   Active Set selection logic.

### CYCLE 05: Dynamics Engine (MD/kMC & OTF)
*   **Goal**: Implement the execution engine for MD and kMC with On-the-Fly learning.
*   **Features**:
    *   `Dynamics` component implementation.
    *   LAMMPS interface via `lammps` python module or subprocess.
    *   Hybrid potential setup (`pair_style hybrid/overlay`).
    *   Uncertainty monitoring (`fix halt` triggered by extrapolation grade).
    *   EON interface for kMC (optional/stub if EON is complex to install in CI).

### CYCLE 06: Validation & Full Orchestration
*   **Goal**: Finalise the system with quality assurance and full loop integration.
*   **Features**:
    *   `Validator` component implementation (Phonons, Elasticity, EOS).
    *   Full integration of all components in `Orchestrator`.
    *   CLI commands for running the full pipeline.
    *   Final system testing and documentation.

## 6. Test Strategy

### 6.1 Unit Testing
*   Each component (Generator, Oracle, etc.) will be tested in isolation using mocks.
*   Pydantic models will be tested for validation logic.
*   Utils will be tested for edge cases.

### 6.2 Integration Testing
*   **Mock Integration**: Verify the `Orchestrator` can drive the loop using Mock components (Cycle 01).
*   **Component Integration**: Verify that `Trainer` can read `Dataset` output, `Oracle` can read `Generator` output, etc.

### 6.3 End-to-End (E2E) Testing
*   Run a full cycle using a small test dataset (Mock DFT or fast tight-binding).
*   Verify the system completes without crashing and produces a valid potential file.
*   **CI Strategy**: Use "Mock Mode" for CI where heavy calculations (DFT, Training) are simulated or done on tiny systems.
