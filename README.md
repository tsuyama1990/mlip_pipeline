# PYACEMAKER (MLIP Pipeline)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Democratizing SOTA Machine Learning Potentials for Everyone.**

PYACEMAKER is a fully automated, "Zero-Config" pipeline that orchestrates the construction of Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It bridges the gap between high-accuracy Density Functional Theory (DFT) and large-scale Molecular Dynamics (MD), allowing non-experts to generate robust, physics-informed potentials in days, not months.

## Key Features

*   **Zero-Config Workflow**: Define your material system in a single `config.yaml` and let the system handle the rest (sampling, calculating, training).
*   **Physics-Informed Robustness**: Enforces safety constraints via Hybrid Potentials (ACE + ZBL/LJ), preventing unphysical behavior like nuclear fusion in extrapolation regions.
*   **Active Learning with "Watchdogs"**: Automatically detects high-uncertainty configurations during MD simulations, halts the run, and triggers a self-healing learning loop.
*   **Self-Healing Oracle**: A DFT interface that automatically detects convergence failures and adjusts parameters to recover without human intervention.
*   **Multi-Scale Integration**: Seamlessly bridges atomic MD with long-timescale Adaptive Kinetic Monte Carlo (aKMC) to study diffusion and ordering phenomena.

## Architecture Overview

The system operates as an autonomous loop, driven by an **Orchestrator** that coordinates specialized agents.

```mermaid
graph TD
    subgraph "Control Plane"
        Orch[Orchestrator]
    end

    subgraph "Exploration & Detection"
        SG[Structure Generator]
        DE[Dynamics Engine]
    end

    subgraph "Ground Truth"
        Oracle[Oracle (DFT)]
    end

    subgraph "Learning"
        Trainer[Trainer (Pacemaker)]
    end

    Orch --> SG
    Orch --> DE
    DE -- "High Uncertainty" --> Oracle
    Oracle -- "Data" --> Trainer
    Trainer -- "Potential" --> DE
```

## Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended package manager)
*   **Docker** or **Singularity** (for running external engines like LAMMPS/QE in isolation)
*   **Quantum Espresso** / **VASP** (for DFT calculations)
*   **LAMMPS** (for MD simulations)

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install Dependencies**
    We use `uv` for fast dependency management.
    ```bash
    uv sync
    ```

3.  **Configure Environment**
    Copy the example configuration.
    ```bash
    cp .env.example .env
    # Edit .env to point to your DFT/LAMMPS executables
    ```

## Usage

### Quick Start
To run a demonstration cycle (using Mock components):

```bash
uv run mlip-pipeline run config/examples/quickstart.yaml
```

### Full Production Run
1.  Edit `config.yaml` to define your system (e.g., `elements: [Fe, Pt]`).
2.  Launch the pipeline:
    ```bash
    uv run mlip-pipeline run config.yaml
    ```

## Development Workflow

The project follows a strict cycle-based development plan.

### Running Tests
```bash
uv run pytest
```

### Linting & Type Checking
We enforce strict code quality.
```bash
uv run ruff check
uv run mypy .
```

## Project Structure

```ascii
src/mlip_autopipec/
├── config/                  # Configuration Logic
├── domain_models/           # Pydantic Data Models
├── interfaces/              # Core Protocols
├── orchestration/           # Main Loop Logic
├── structure_generation/    # Adaptive Exploration
├── oracle/                  # DFT & Self-Healing
├── training/                # Pacemaker Integration
├── dynamics/                # LAMMPS & OTF Loop
├── validation/              # Physics Checks
└── utils/                   # Logging & I/O

dev_documents/
├── system_prompts/          # Architecture & Cycle Specs
└── tutorials/               # User Guides
```

## License

MIT License. See `LICENSE` for details.
