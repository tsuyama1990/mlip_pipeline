# PYACEMAKER

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Automated Active Learning for Machine Learning Interatomic Potentials.**

## Overview

**What:** PYACEMAKER is a fully automated, "Zero-Config" workflow for constructing robust Machine Learning Interatomic Potentials (MLIPs), specifically focusing on Atomic Cluster Expansion (ACE).

**Why:** Developing high-fidelity potentials traditionally involves manual, error-prone cycles of DFT calculations and fitting. PYACEMAKER automates this process using an adaptive active learning loop, democratizing access to state-of-the-art potentials for materials science.

## Features

-   **Zero-Config Workflow**: Define your material system and target accuracy in a single YAML file.
-   **Adaptive Structure Generation**: Automatically generates diverse structures (bulk, surfaces) with intelligent sampling policies.
-   **High-Fidelity Oracle**: Seamless integration with **Quantum Espresso** for accurate DFT labeling of energy, forces, and stress.
-   **Self-Healing DFT**: Robust error handling that automatically retries failed calculations with adjusted parameters (e.g., mixing beta, smearing).
-   **Periodic Embedding**: Intelligent extraction of local clusters from large MD snapshots for efficient QM/MM-style labeling.
-   **Active Learning**: Automatically explores configuration space and selects informative structures using D-optimality (MaxVol).
-   **Physics-Informed Robustness**: Implements Delta Learning to fit the residual energy against a physical baseline (Lennard-Jones/ZBL).
-   **Modular Architecture**: Extensible design with swappable components.

## Requirements

-   Python >= 3.12
-   [UV](https://github.com/astral-sh/uv) (recommended)
-   Quantum Espresso (`pw.x`) installed and in PATH (for production runs).
-   Pacemaker (`pace_train`, `pace_activeset`) installed and in PATH (for training).

## Installation

```bash
git clone https://github.com/your-org/mlip-pipeline.git
cd mlip-pipeline
uv sync
```

## Usage

1.  **Create a Configuration File** (`config.yaml`):

    ```yaml
    workdir: ./output_experiment
    max_cycles: 5
    logging_level: INFO

    components:
      generator:
        name: adaptive
        element: Si
        crystal_structure: diamond
        n_structures: 10
      oracle:
        name: qe
        kspacing: 0.1
        ecutwfc: 30.0
        ecutrho: 150.0
        mixing_beta: 0.7
      trainer:
        name: pacemaker
        cutoff: 5.0
        basis_size: 500
        physics_baseline:
          type: lj
          params:
            sigma: 2.5
            epsilon: 0.1
      dynamics:
        name: mock
        selection_rate: 0.5
      validator:
        name: mock
    ```

2.  **Run the Pipeline**:

    ```bash
    uv run python src/mlip_autopipec/main.py run config.yaml
    ```

## Architecture

```ascii
src/mlip_autopipec/
├── components/         # Pluggable components (Generator, Oracle, Trainer, etc.)
│   ├── trainer/        # Pacemaker Integration & Active Set Logic
│   ├── oracle/         # DFT Engines (QE, VASP) & Healing Logic
│   └── ...
├── core/               # Core logic (Orchestrator, Dataset, State)
├── domain_models/      # Pydantic data models (Structure, Potential, Config)
├── interfaces/         # Abstract base classes
├── utils/              # Utilities (Logging, etc.)
└── main.py             # CLI Entry Point
```

## Roadmap

-   [x] Cycle 01: Core Framework & Mock Components
-   [x] Cycle 02: Structure Generator (ASE integration)
-   [x] Cycle 03: Oracle (Quantum Espresso, Self-Healing, Embedding)
-   [x] Cycle 04: Trainer (Pacemaker integration)
-   [ ] Cycle 05: Dynamics Engine (LAMMPS/EON integration)
-   [ ] Cycle 06: Validation & Full Orchestration
