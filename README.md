# PYACEMAKER: Automated MLIP Construction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/mlip-pipeline)

**PYACEMAKER** (Python Atomic Cluster Expansion Maker) is an automated system designed to democratize the creation of high-quality Machine Learning Interatomic Potentials (MLIP). By orchestrating the entire active learning loop—from structure generation and DFT calculation to potential fitting and validation—it allows researchers to develop "State-of-the-Art" potentials with minimal manual intervention.

## Key Features

-   **Zero-Config Workflow**: Define your material system in a single YAML file and let the system handle the rest. No complex scripting required.
-   **Active Learning with Uncertainty**: Drastically reduces DFT costs (by >90%) by only calculating structures where the model is uncertain ($\gamma$ metric).
-   **Physics-Informed Robustness**: Enforces a physical baseline (Lennard-Jones/ZBL) to ensure simulation stability even in high-energy regimes where data is scarce.
-   **Self-Healing Oracle**: Automatically recovers from DFT convergence failures by adjusting calculation parameters on the fly.
-   **Automated Validation**: Every generated potential is rigorously tested for physical stability (Elastic constants, Phonons) before deployment.

## Architecture Overview

The system follows a hub-and-spoke architecture where a central Orchestrator manages the data flow between specialized components.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Init| Gen[Structure Generator]
    Orch -->|Loop| Dyn[Dynamics Engine]

    subgraph Active Learning Loop
        Gen -->|Candidate Structures| Oracle[Oracle (DFT)]
        Dyn -->|High Uncertainty Structures| Oracle
        Oracle -->|Labeled Data| DB[(Dataset)]
        DB -->|Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|potential.yace| Val[Validator]
        Val -- Pass --> Orch
        Val -- Fail --> Gen
    end

    Dyn -->|MD/kMC Trajectory| Results[Simulation Results]
    Trainer -->|Physics Baseline| Dyn
```

## Prerequisites

-   **Python**: Version 3.12 or higher.
-   **Package Manager**: `uv` (recommended) or `pip`.
-   **External Tools** (Optional but recommended for full functionality):
    -   `Quantum Espresso` (pw.x) for DFT calculations.
    -   `LAMMPS` (lmp) for MD simulations.
    -   `Pacemaker` (pace_train) for potential fitting.

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install Dependencies (using uv)**
    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Sync dependencies
    uv sync
    ```

3.  **Activate Virtual Environment**
    ```bash
    source .venv/bin/activate
    ```

## Usage

To start a new project, create a configuration file (e.g., `config.yaml`) and run the pipeline.

### Quick Start

1.  **Create a Config File**
    ```yaml
    workdir: "runs/fe_pt_demo"
    max_cycles: 5
    generator:
      type: "random"
      composition: {"Fe": 0.5, "Pt": 0.5}
    oracle:
      type: "mock"  # Use 'espresso' for real DFT
    trainer:
      type: "mock"  # Use 'pacemaker' for real training
    dynamics:
      type: "mock"  # Use 'lammps' for real MD
    ```

2.  **Run the Pipeline**
    ```bash
    mlip-pipeline run config.yaml
    ```

## Development Workflow

We follow the AC-CDD (Architectural-Core Cycle Driven Development) methodology.

-   **Running Tests**:
    ```bash
    pytest
    ```

-   **Linting & Formatting**:
    ```bash
    ruff check .
    ruff format .
    ```

-   **Type Checking**:
    ```bash
    mypy .
    ```

## Project Structure

```ascii
mlip-pipeline/
├── dev_documents/        # Specifications & UAT Plans
├── src/
│   └── mlip_autopipec/
│       ├── components/   # Core modules (Generator, Oracle, etc.)
│       ├── core/         # Orchestrator & Dataset logic
│       ├── domain_models/# Pydantic schemas
│       ├── interfaces/   # Abstract Base Classes
│       └── main.py       # CLI Entry point
├── tests/                # Unit & Integration tests
├── tutorials/            # Jupyter Notebooks for User Training
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
