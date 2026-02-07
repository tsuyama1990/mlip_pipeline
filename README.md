# PYACEMAKER (MLIP Pipeline)

![Status](https://img.shields.io/badge/Status-Development-yellow)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Democratizing Atomistic Simulations: From Zero to State-of-the-Art in One Config.**

PYACEMAKER is an automated pipeline for constructing and operating high-efficiency Machine Learning Interatomic Potentials (MLIPs). It leverages the **ACE (Atomic Cluster Expansion)** framework via **Pacemaker** to deliver potentials that are both accurate (DFT-grade) and fast (MD-grade). Designed for materials scientists, it automates the complex loop of structure generation, active learning, and validation, requiring only a single configuration file to run.

## Key Features

*   **Zero-Config Automation**: Define your material system in `config.yaml` and let the orchestrator handle the rest. No Python scripting required.
*   **Active Learning Efficiency**: Uses uncertainty-based sampling and D-optimality (Active Set) to minimise expensive DFT calculations.
*   **Physical Robustness**: Hybrid potentials (ACE + ZBL/LJ) ensure stability even in high-energy regimes, preventing simulation crashes.
*   **Self-Healing Oracle**: Automatically recovers from DFT convergence failures (Quantum Espresso) by adjusting parameters on the fly.
*   **Bridge to Scales**: Seamlessly integrates Molecular Dynamics (LAMMPS) and Kinetic Monte Carlo (EON) to span time scales from femtoseconds to seconds.
*   **Mock Mode**: Fully implemented Mock components (Oracle, Trainer, Dynamics, etc.) for testing the workflow without external binaries.

## Architecture Overview

The system is built on a modular, cycle-based architecture.

```mermaid
graph TD
    User[User] -->|Config (YAML)| Orchestrator
    Orchestrator -->|Control| Generator[Structure Generator]
    Orchestrator -->|Control| Oracle[Oracle / DFT Manager]
    Orchestrator -->|Control| Trainer[Trainer / Pacemaker]
    Orchestrator -->|Control| Dynamics[Dynamics Engine / LAMMPS]
    Orchestrator -->|Control| Validator[Validator]

    subgraph "Data Store"
        Dataset[(Dataset .pckl)]
        Potential[(Potential .yace)]
    end

    Generator -->|Candidate Structures| Oracle
    Dynamics -->|Uncertain Structures| Oracle
    Oracle -->|Labelled Structures| Dataset
    Dataset --> Trainer
    Trainer -->|New Potential| Potential
    Potential --> Dynamics
    Potential --> Validator
    Validator -->|Validation Report| Orchestrator
```

## Prerequisites

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Tools**:
    *   **Quantum Espresso** (`pw.x`) for DFT calculations (optional if using pre-calculated data).
    *   **LAMMPS** (`lmp_serial` or `lmp_mpi`) for Molecular Dynamics.
    *   **Pacemaker** (Python library) for potential fitting.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install dependencies:**
    We recommend using `uv` for fast, reliable dependency management.
    ```bash
    uv sync
    ```
    Or with pip:
    ```bash
    pip install -e .
    ```

## Usage

**Run the pipeline:**
```bash
mlip-pipeline run config.yaml
```

**Quick Start (Mock Mode):**
To test the workflow without external binaries, create a mock configuration:

```yaml
workdir: "experiments/mock_run"
oracle:
  type: "mock"
  noise_level: 0.1
trainer:
  type: "mock"
dynamics:
  type: "mock"
generator:
  type: "mock"
validator:
  type: "mock"
selector:
  type: "mock"
```

Then run:
```bash
mlip-pipeline run config_mock.yaml
```

## Development Workflow

We follow the **AC-CDD (Architect-Coder-Auditor)** methodology.

**Run Tests:**
```bash
uv run pytest
```

**Linting & Formatting:**
Strict code quality is enforced via `ruff` and `mypy`.
```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```ascii
src/
└── mlip_autopipec/
    ├── domain_models/      # Pydantic data models
    ├── interfaces/         # Abstract Base Classes
    ├── infrastructure/     # Concrete classes (Mocks, etc.)
    ├── factory.py          # Component Factory
    └── main.py             # CLI entry point

dev_documents/
├── system_prompts/         # Cycle-based specifications
├── FINAL_UAT.md            # Tutorial plan
└── SYSTEM_ARCHITECTURE.md  # Detailed system design
```

## License

This project is licensed under the MIT License.
