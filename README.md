# PyAcemaker: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAcemaker** is a "Zero-Config" automated system for constructing State-of-the-Art Machine Learning Interatomic Potentials (MLIP). By leveraging the **Pacemaker** (Atomic Cluster Expansion) engine and an intelligent **Active Learning** loop, it democratises access to high-accuracy atomistic simulations, allowing users to generate robust potentials without deep expertise in computational physics.

## Key Features

-   **Zero-Config Workflow**: From a single `config.yaml`, the system handles structure generation, DFT calculations, training, and validation autonomously.
-   **Data Efficiency**: Uses Active Learning and D-Optimality to achieve production-level accuracy with **1/10th of the DFT cost** of random sampling.
-   **Physics-Informed Robustness**: Guarantees simulation stability by enforcing physical baselines (ZBL/LJ) and monitoring extrapolation grades ($\gamma$) in real-time.
-   **Self-Healing Oracle**: Automatically recovers from DFT convergence failures, ensuring the pipeline never stalls due to numerical instability.
-   **Multi-Scale Exploration**: Integrates MD (LAMMPS) and kMC (EON) to explore both fast thermal vibrations and slow rare events.

## Architecture Overview

PyAcemaker follows a modular architecture orchestrated by a central Python controller.

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]

    subgraph "Core Loop"
        Orch --> SG[Structure Generator]
        Orch --> DE[Dynamics Engine]

        SG -->|Candidates| DB[(Database)]
        DE -->|High Uncertainty Structures| DB

        DB -->|Selected Structures| Oracle[Oracle (DFT)]
        Oracle -->|Labelled Data| DB

        DB -->|Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|Potential (yace)| Val[Validator]

        Val -- Pass --> DE
        Val -- Fail --> SG
    end
```

## Prerequisites

-   **Python 3.11+**
-   **uv** (Recommended for dependency management)
-   **Quantum Espresso** (`pw.x`) installed and accessible in `$PATH`.
-   **LAMMPS** (with USER-PACE package installed).
-   **Pacemaker** library.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/mlip_autopipec.git
    cd mlip_autopipec
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
    Alternatively, using pip:
    ```bash
    pip install -e .[dev]
    ```

3.  **Configure Environment:**
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    ```
    Edit `config.yaml` to point to your pseudopotential directory and executables.

## Usage

### Quick Start

To start a new potential generation project:

```bash
# Initialize the project
uv run mlip-auto init --name "Silicon_Project"

# Validate configuration
uv run mlip-auto validate

# Run the automated pipeline
uv run mlip-auto run
```

### Monitoring

The system generates a `report.html` in the project directory, updating in real-time with training curves, validation metrics (Phonon/EOS), and active learning status.

## Development Workflow

We follow the **AC-CDD** (Active Cycle - Component Driven Development) methodology.

1.  **Running Tests:**
    ```bash
    uv run pytest
    ```

2.  **Linting & Formatting:**
    This project enforces strict code quality.
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

## Project Structure

```text
.
├── src/
│   └── mlip_autopipec/
│       ├── config/         # Pydantic Schemas
│       ├── dft/            # Oracle & QE Interface
│       ├── dynamics/       # LAMMPS & EON Runners
│       ├── generator/      # Structure Generation
│       ├── orchestration/  # Workflow Logic
│       ├── training/       # Pacemaker Wrapper
│       └── validation/     # Physics Checks
├── dev_documents/          # System Architecture & Specs
├── tests/                  # Pytest Suite
└── pyproject.toml          # Dependencies & Linter Config
```

## License

This project is licensed under the MIT License.
