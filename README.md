# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system for constructing and operating Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It democratizes the creation of "State-of-the-Art" potentials by automating the entire lifecycle: from adaptive structure sampling (MD/MC) and self-healing DFT calculations (Quantum Espresso) to active learning and uncertainty-driven refinement.

Designed for materials scientists, it turns a complex, multi-week manual workflow into a "Zero-Config" operation defined by a single YAML file.

## Key Features

*   **Zero-Configuration Workflow**: Initiate complex active learning campaigns with a single configuration file. No Python scripting required.
*   **Active Learning with Uncertainty**: Automatically detects "unknown" physics during simulations using extrapolation grades ($\gamma$) and triggers targeted retraining loops ("Halt & Diagnose").
*   **Physics-Informed Robustness**: Prevents unphysical explosions by combining ACE with robust baseline potentials (LJ/ZBL) via Delta Learning.
*   **Self-Healing Oracle**: Automatically recovers from DFT convergence failures (SCF divergence) by adjusting mixing parameters and smearing temperatures.
*   **Multi-Scale Simulation**: Seamlessly bridges the gap between fast Molecular Dynamics (LAMMPS) and long-term evolution (Adaptive Kinetic Monte Carlo via EON).

## Architecture Overview

PYACEMAKER uses an Orchestrator-Agent pattern to manage the scientific workflow.

```mermaid
graph TD
    subgraph Control Plane
        Orchestrator[Orchestrator]
        Config[Configuration (YAML)]
    end

    subgraph "Exploration & Dynamics"
        Gen[Structure Generator]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Watchdog{Uncertainty Watchdog}
    end

    subgraph "Learning & Verification"
        Oracle[Oracle (DFT/QE)]
        Trainer[Trainer (Pacemaker)]
        Validator[Validator]
    end

    Config --> Orchestrator
    Orchestrator --> Gen
    Orchestrator --> Dyn
    Orchestrator --> Oracle
    Orchestrator --> Trainer
    Orchestrator --> Validator

    Gen -- "Candidate Structures" --> Oracle
    Dyn -- "Halted Structure" --> Gen
    Dyn -- "Trajectory" --> Watchdog
    Watchdog -- "High Uncertainty" --> Orchestrator

    Oracle -- "Labeled Data (E, F, S)" --> Trainer
    Trainer -- "Potential (.yace)" --> Dyn
    Trainer -- "Potential (.yace)" --> Validator
    Validator -- "Quality Report" --> Orchestrator
```

## Prerequisites

*   **Python**: 3.11 or higher
*   **External Engines**:
    *   `lammps` (with USER-PACE package)
    *   `pw.x` (Quantum Espresso) for DFT
    *   `pace_train` / `pace_activeset` (Pacemaker) for training
    *   `eonclient` (EON) for kMC (optional)
*   **Package Manager**: `uv` (recommended) or `pip`

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install dependencies**:
    Using `uv` (recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -e .
    pip install -r requirements-dev.txt
    ```

3.  **Prepare the environment**:
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    ```
    Ensure external binaries are in your `$PATH`.

## Usage

### Quick Start (Mock Mode)
To verify the system without running heavy physics codes, use the "Mock Mode".

1.  Edit `config.yaml` and set `mode: mock`.
2.  Run the pipeline:
    ```bash
    python src/mlip_autopipec/main.py --config config.yaml
    ```
    This will generate dummy potentials and logs in the `output/` directory.

### Production Run
For a real active learning campaign:

1.  Set `mode: production` in `config.yaml`.
2.  Configure your DFT parameters (pseudopotentials, k-spacing).
3.  Run the system (recommended to run in background or via SLURM):
    ```bash
    nohup python src/mlip_autopipec/main.py --config config.yaml > run.log 2>&1 &
    ```

## Development Workflow

This project follows a cycle-based development plan (Cycle 01 - Cycle 08).

*   **Running Tests**:
    ```bash
    pytest
    ```

*   **Linting & Formatting**:
    We enforce strict code quality using `ruff` and `mypy`.
    ```bash
    ruff check .
    mypy .
    ```

## Project Structure

```
.
├── config.yaml               # User configuration
├── pyproject.toml            # Dependencies and tools
├── src/
│   └── mlip_autopipec/
│       ├── core/             # Orchestrator, Config, Logging
│       ├── components/       # Generator, Oracle, Trainer, Dynamics
│       ├── domain_models/    # Pydantic data models
│       └── utils/            # Drivers for LAMMPS, Pacemaker, etc.
├── tests/                    # Unit and Integration tests
└── dev_documents/            # Specifications and UAT plans
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
