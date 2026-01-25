# PyAcemaker: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAcemaker** is a "Zero-Config" automated system for constructing "State-of-the-Art" Machine Learning Interatomic Potentials (MLIPs). Built on top of the **Pacemaker** (Atomic Cluster Expansion) engine, it utilizes an autonomous Active Learning workflow to generate physically robust potentials with minimal user intervention.

> **Goal:** Democratize high-accuracy atomistic simulations by removing the need for deep expertise in data science and computational physics.

## Key Features

*   **Zero-Config Automation**: A single `config.yaml` drives the entire pipeline—from initial structure generation to final potential deployment.
*   **Active Learning (Data Efficiency)**: Reduces DFT costs by >90% using D-Optimality (Active Set) selection and real-time uncertainty monitoring ($\gamma$-watchdog).
*   **Physics-Informed Robustness**: Enforces **Hybrid Potentials** (ACE + ZBL/LJ) to prevent physical violations (e.g., core collapse) and ensures stability via Phonon and Elasticity validation.
*   **Self-Healing Oracle**: Automated DFT execution (Quantum Espresso) with built-in error recovery for SCF convergence failures.
*   **Scalability**: Modular design supporting deployment from local workstations to HPC clusters.

## Architecture Overview

PyAcemaker follows a Hub-and-Spoke architecture with a central **Orchestrator** managing specialized agents.

```mermaid
graph TD
    User[User / Config.yaml] --> Orch(Orchestrator)

    subgraph "Core Loop (Active Learning)"
        Orch --> Dyn(Dynamics Engine<br/>LAMMPS / EON)
        Dyn -- "Halted (High Uncertainty)" --> Sel(Selection Module)
        Sel -- "Candidates (Embedded)" --> Oracle(Oracle / DFT<br/>Quantum Espresso)
        Oracle -- "Labeled Data" --> Train(Trainer<br/>Pacemaker)
        Train -- "New Potential" --> Val(Validator)
        Val -- "Verified Potential" --> Orch
        Orch -- "Deploy" --> Dyn
    end

    subgraph "Auxiliary Modules"
        Gen(Structure Generator) --> Sel
        Dash(Dashboard / Reporter)
    end
```

## Prerequisites

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Tools**:
    *   **Quantum Espresso** (`pw.x`) for DFT calculations.
    *   **LAMMPS** (`lmp`) with USER-PACE package for MD simulations.
    *   **Pacemaker** (`pace_train`, `pace_collect`) for potential training.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies using `uv`**:
    ```bash
    uv sync
    ```
    *Alternatively, with pip:* `pip install -e .`

3.  **Set up environment**:
    Copy the example environment file and configure paths to your executables.
    ```bash
    cp .env.example .env
    # Edit .env to set PW_COMMAND, LMP_COMMAND, etc.
    ```

## Usage

### Quick Start

1.  **Initialize a project**:
    Generate a default configuration file.
    ```bash
    mlip-auto init
    ```

2.  **Configure**:
    Edit `config.yaml` to specify your target system (e.g., elements, crystal structure).

3.  **Run the pipeline**:
    ```bash
    mlip-auto run --config config.yaml
    ```

4.  **Monitor**:
    Open `report.html` in your browser to watch the active learning progress and validation metrics.

## Development Workflow

This project follows the **AC-CDD** (Architecturally-Constrained Cycle-Driven Development) methodology.

### Running Tests
We use `pytest` for unit and integration testing.
```bash
uv run pytest
```

### Linting & Formatting
We strictly enforce code quality using `ruff` and `mypy`.
```bash
uv run ruff check .
uv run mypy .
```

### Project Structure

```ascii
src/mlip_autopipec/
├── orchestration/          # Main workflow logic
├── phases/                 # Workflow steps (DFT, Training, etc.)
│   ├── dft/                # Oracle (Quantum Espresso)
│   ├── training/           # Trainer (Pacemaker)
│   ├── dynamics/           # MD Engine (LAMMPS)
│   ├── selection/          # Active Learning Logic
│   └── validation/         # Physics Validation
├── config/                 # Pydantic Schemas
└── generator/              # Structure Generator
dev_documents/              # System Prompts & Specifications
tests/                      # Test Suite
```

## License

MIT License. See `LICENSE` for details.
