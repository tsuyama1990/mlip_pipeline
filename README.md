# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIPs). By orchestrating a closed-loop active learning process, it enables materials scientists to generate "State-of-the-Art" ACE potentials with **Zero-Config** effort, minimizing expensive DFT calculations while ensuring physical robustness.

## Key Features

-   **Zero-Config Workflow**: Define your material system in a single `config.yaml` file. The system handles structure generation, DFT submission, training, and validation automatically.
-   **Data Efficiency**: Uses **Active Learning** and D-Optimality (MaxVol) to select only the most informative structures, reducing DFT costs by up to 90%.
-   **Physics-Informed Robustness**: Implements **Delta Learning** (learning the difference from a physical baseline like LJ/ZBL) to guarantee stability even in far-from-equilibrium regimes.
-   **Self-Healing Oracle**: Automatically detects and corrects DFT convergence failures (Quantum Espresso) by adjusting mixing parameters and algorithms.
-   **Hybrid Simulation**: Seamlessly integrates Molecular Dynamics (LAMMPS) and Adaptive Kinetic Monte Carlo (EON) to explore both fast thermal fluctuations and slow rare events.

## Architecture Overview

PYACEMAKER uses a Hub-and-Spoke architecture centered around an intelligent **Orchestrator**.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch --> Gen[Structure Generator]
    Orch --> Oracle[Oracle (DFT)]
    Orch --> Trainer[Trainer (Pacemaker)]
    Orch --> Dyn[Dynamics (MD/kMC)]
    Orch --> Val[Validator]

    Gen -->|Candidates| Oracle
    Dyn -->|Uncertainty Halt| Oracle
    Oracle -->|Labeled Data| Trainer
    Trainer -->|Potential .yace| Dyn
    Trainer -->|Potential .yace| Val
    Val -->|Report| User
```

## Prerequisites

-   **Python 3.12+**
-   **uv** (Recommended package manager)
-   **Quantum Espresso** (`pw.x`) - For Oracle
-   **LAMMPS** (`lmp`) with `USER-PACE` package - For Dynamics
-   **Pacemaker** (`pace_train`) - For Training

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Set up environment**:
    ```bash
    cp .env.example .env
    # Edit .env to set paths to QE, LAMMPS, etc.
    ```

## Usage

### Quick Start

1.  **Create a configuration file** (`config.yaml`):
    ```yaml
    workdir: ./output
    max_cycles: 3
    system:
      element: Fe
    components:
      generator: {type: "random", n: 10}
      oracle: {type: "qe", command: "pw.x"}
      trainer: {type: "pacemaker"}
      dynamics: {type: "lammps"}
    ```

2.  **Run the pipeline**:
    ```bash
    uv run python -m mlip_autopipec run config.yaml
    ```

3.  **View the report**:
    Open `output/report.html` in your browser to see the learning curves and validation results.

## Development Workflow

We follow the AC-CDD methodology with 6 implementation cycles.

### Running Tests
```bash
uv run pytest
```

### Linting & Type Checking
This project enforces strict code quality.
```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```ascii
.
├── dev_documents/          # Specs, UATs, and Architecture Docs
├── src/
│   └── mlip_autopipec/
│       ├── components/     # Generator, Oracle, Trainer, Dynamics, Validator
│       ├── core/           # Orchestrator, Dataset, State
│       ├── domain_models/  # Pydantic Models (Config, Structure)
│       └── interfaces/     # Base Classes
├── tests/                  # Unit and Integration Tests
├── tutorials/              # Jupyter Notebooks for UAT
└── pyproject.toml          # Project Configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
