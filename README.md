# PYACEMAKER

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

**Automated Active Learning for Machine Learning Interatomic Potentials.**

## Overview

**What:** PYACEMAKER is a fully automated, "Zero-Config" workflow for constructing robust Machine Learning Interatomic Potentials (MLIPs), specifically focusing on Atomic Cluster Expansion (ACE).

**Why:** Developing high-fidelity potentials traditionally involves manual, error-prone cycles of DFT calculations and fitting. PYACEMAKER automates this process using an adaptive active learning loop, democratizing access to state-of-the-art potentials for materials science.

## Features

-   **Zero-Config Workflow**: Define your material system and target accuracy in a single YAML file.
-   **Active Learning**: Automatically explores configuration space and selects informative structures for labeling.
-   **Modular Architecture**: Extensible design with swappable components for Generation, Labeling (Oracle), Training, Dynamics, and Validation.
-   **Robustness**: Built-in validation and error handling to ensure physical stability.
-   **Mock Mode**: Includes a full mock implementation for testing and development without heavy compute requirements.

## Requirements

-   Python >= 3.12
-   [UV](https://github.com/astral-sh/uv) (recommended for dependency management)

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
        type: mock
        n_structures: 10
      oracle:
        type: mock
      trainer:
        type: mock
      dynamics:
        type: mock
        selection_rate: 0.5
      validator:
        type: mock
    ```

2.  **Run the Pipeline**:

    ```bash
    uv run python main.py run config.yaml
    ```

## Architecture

```ascii
src/mlip_autopipec/
├── components/         # Pluggable components (Generator, Oracle, Trainer, etc.)
├── core/               # Core logic (Orchestrator, Dataset, State)
├── domain_models/      # Pydantic data models (Structure, Potential, Config)
├── interfaces/         # Abstract base classes
├── utils/              # Utilities (Logging, etc.)
└── main.py             # CLI Entry Point
```

## Roadmap

-   [x] Cycle 01: Core Framework & Mock Components
-   [ ] Cycle 02: Structure Generator (ASE/Pymatgen integration)
-   [ ] Cycle 03: Oracle (Quantum Espresso integration)
-   [ ] Cycle 04: Trainer (Pacemaker integration)
-   [ ] Cycle 05: Dynamics Engine (LAMMPS/EON integration)
-   [ ] Cycle 06: Validation & Full Orchestration
