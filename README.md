# PYACEMAKER (MLIP Pipeline)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Democratizing SOTA Machine Learning Potentials for Everyone.**

PYACEMAKER is a fully automated, "Zero-Config" pipeline that orchestrates the construction of Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It bridges the gap between high-accuracy Density Functional Theory (DFT) and large-scale Molecular Dynamics (MD), allowing non-experts to generate robust, physics-informed potentials in days, not months.

## Key Features

*   **Zero-Config Workflow**: Define your material system in a single `config.yaml` and let the system handle the rest (sampling, calculating, training).
*   **Physics-Informed Robustness**: Enforces safety constraints via Hybrid Potentials (ACE + ZBL/LJ).
*   **Active Learning**: Automatically explores the chemical space to build robust training sets.
*   **Modular Architecture**: Plug-and-play components for Structure Generation, DFT (Oracle), and Training.

## Architecture Overview

The system operates as an autonomous loop, driven by an **Orchestrator** that coordinates specialized agents.

```mermaid
graph TD
    subgraph "Control Plane"
        Orch[Orchestrator]
    end

    subgraph "Exploration"
        SG[Structure Generator]
    end

    subgraph "Ground Truth"
        Oracle[Oracle (DFT)]
    end

    subgraph "Learning"
        Trainer[Trainer (Pacemaker)]
    end

    Orch --> SG
    SG --> Oracle
    Oracle --> Trainer
```

## Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended package manager)

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

3.  **Activate Environment**
    ```bash
    source .venv/bin/activate
    ```

## Usage

### Quick Start
To run a demonstration cycle (using Mock components):

1.  **Create a Configuration File** (`config.yaml`):
    ```yaml
    project_name: "my_demo"
    execution_mode: "mock"
    cycles: 3
    dft:
      calculator: "lj"
      kpoints_density: 0.04
      encut: 500.0
    training:
      potential_type: "ace"
      cutoff: 5.0
      max_degree: 1
    exploration:
      strategy: "random"
      num_candidates: 10
      supercell_size: 2
    ```

2.  **Run the Pipeline**:
    ```bash
    mlip-pipeline run config.yaml
    ```

3.  **Check Output**:
    The system will generate log messages indicating the progress of the active learning loop and create "potential" artifacts in the current directory.

## Development Workflow

### Running Tests
```bash
uv run pytest
```

### Linting & Type Checking
We enforce strict code quality.
```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```ascii
src/mlip_autopipec/
├── config/                  # Configuration Logic
├── domain_models/           # Pydantic Data Models
├── interfaces/              # Core Protocols
├── orchestration/           # Main Loop Logic
└── utils/                   # Logging & I/O
```

## License

MIT License. See `LICENSE` for details.
