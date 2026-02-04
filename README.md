# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an automated system for constructing robust Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It democratizes access to state-of-the-art potential generation by providing a "Zero-Config" workflow that autonomously iterates through structure generation, DFT calculation, training, and validation.

## Key Features

-   **Zero-Config Automation**: Go from chemical composition to a fully trained `.yace` potential with a single YAML file.
-   **Physics-Informed Robustness**: Automatically incorporates Lennard-Jones/ZBL baselines to prevent non-physical extrapolation and simulation crashes.
-   **Self-Healing Oracle**: Automated DFT calculations (Quantum Espresso/VASP) with built-in error recovery for SCF convergence failures.
-   **Active Learning**: Intelligent structure generation using adaptive policies (MD, Monte Carlo, Defects) to sample relevant configuration spaces efficiently.
-   **Hybrid Simulation**: Seamlessly integrates MD (LAMMPS) for fast dynamics and Adaptive kMC (EON) for long-timescale phenomena like ordering.

## Architecture Overview

The system is orchestrated by a central Python controller that manages a loop of specialized workers.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Manage| Loop{Active Learning Loop}

    subgraph "Core Modules"
        Explorer[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Val[Validator]
    end

    Loop -->|1. Request Structures| Explorer
    Explorer -->|Candidate Structures| Loop

    Loop -->|2. Request Data| Oracle
    Oracle -->|Labeled Data (E, F, S)| Loop

    Loop -->|3. Train| Trainer
    Trainer -->|Potential (.yace)| Loop

    Loop -->|4. Simulate & Check| Dyn
    Dyn -->|Uncertainty / Halt| Loop

    Loop -->|5. Verify| Val
    Val -->|Pass/Fail| Loop

    Loop -->|Final Output| Result[Production Potential]
```

## Prerequisites

-   **Python 3.12+**
-   **uv** (recommended for dependency management)
-   **External Physics Codes** (for Real Mode):
    -   LAMMPS (with USER-PACE package)
    -   Quantum Espresso (pw.x)
    -   Pacemaker (pace_train, pace_activeset)
    -   EON (eonclient)

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Initialize environment**:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env to point to your local LAMMPS/QE executables
    ```

## Usage

To run the pipeline, use the CLI command:

```bash
uv run mlip-pipeline run config.yaml
```

### Quick Start Example

A sample configuration for Fe/Pt is provided in `examples/fe_pt_config.yaml`.

```bash
uv run mlip-pipeline run examples/fe_pt_config.yaml
```

## Development Workflow

The project follows a strict 6-cycle development plan.

-   **Run Tests**:
    ```bash
    uv run pytest
    ```

-   **Linting & Type Checking**:
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

## Project Structure

```ascii
src/mlip_autopipec/
├── config/                  # Configuration Models
├── domain_models/           # Pydantic Data Classes
├── interfaces/              # Protocol Definitions
├── orchestration/           # Main Loop Logic
├── services/                # Business Logic (Trainer, Oracle, etc.)
└── main.py                  # CLI Entrypoint

dev_documents/
├── system_prompts/          # Architectural Specs
└── tutorials/               # Jupyter Notebooks (UAT)
```

## License

This project is licensed under the MIT License.
