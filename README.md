# PYACEMAKER: Automated Active Learning for Interatomic Potentials

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** (internal package: `mlip_autopipec`) is a state-of-the-art system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIP). It automates the entire lifecycle—from initial structure exploration to DFT calculation, potential training, and validation—enabling researchers to generate robust, physics-informed potentials with "Zero Configuration".

## Key Features

*   **Zero-Config Workflow**: Define your material system and desired accuracy in a single YAML file. The system handles parameter tuning for DFT and MD automatically.
*   **Active Learning**: Drastically reduces DFT costs by using uncertainty quantification to select only the most informative structures for training.
*   **Physics-Informed Robustness**: Implements a "Hybrid Potential" strategy (ACE + ZBL/LJ) to prevent unphysical behavior (nuclear fusion) in high-energy regimes.
*   **Self-Healing Oracle**: Automatically detects and recovers from common DFT convergence failures (e.g., charge sloshing) without user intervention.
*   **Multi-Scale Scalability**: Seamlessly integrates Molecular Dynamics (LAMMPS) and Adaptive Kinetic Monte Carlo (EON) to bridge the gap between nanoseconds and hours.

## Architecture Overview

PYACEMAKER follows an Orchestrator pattern where a central controller manages the flow of data between specialized components.

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]
    subgraph "Core Modules"
        Orch --> SG[Structure Generator]
        Orch --> Oracle[Oracle (DFT)]
        Orch --> Trainer[Trainer (Pacemaker)]
        Orch --> DE[Dynamics Engine]
        Orch --> Val[Validator]
    end
    subgraph "External Engines"
        Oracle --> QE[Quantum Espresso / VASP]
        Trainer --> PM[Pacemaker]
        DE --> LAMMPS[LAMMPS]
        DE --> EON[EON (aKMC)]
    end
```

## Prerequisites

*   **Python 3.12+**
*   **uv** (for dependency management)
*   **External Binaries** (for Real Mode):
    *   `lammps` (with USER-PACE package)
    *   `pw.x` (Quantum Espresso) or VASP
    *   `pacemaker` (ACE training tools)
    *   `eonclient` (for aKMC)

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies using uv**:
    ```bash
    uv sync
    ```

3.  **Activate the virtual environment**:
    ```bash
    source .venv/bin/activate
    ```

## Usage

### Quick Start (Mock Mode)
To verify the installation without external heavy dependencies, run the "Hello World" example in Mock Mode:

1.  Initialize a configuration:
    ```bash
    uv run mlip-runner init
    ```

2.  Run the pipeline:
    ```bash
    uv run mlip-runner run config.yaml
    ```

### Production Run
Edit `config.yaml` to set `execution_mode: real` and point to your DFT/LAMMPS binaries, then run:

```bash
uv run mlip-runner run config.yaml
```

## Development Workflow

The project is developed in 8 strict cycles.

*   **Running Tests**:
    ```bash
    uv run pytest
    ```

*   **Linting & Formatting**:
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

## Project Structure

```ascii
.
├── dev_documents/          # Architecture & Specs (AC-CDD)
├── src/
│   └── mlip_autopipec/     # Main Package Source
│       ├── core/           # Orchestrator & Logic
│       ├── generator/      # Structure Generation
│       ├── oracle/         # DFT Interface
│       ├── trainer/        # Pacemaker Interface
│       ├── dynamics/       # LAMMPS/EON Interface
│       └── validator/      # Physics Validation
├── tests/                  # Unit & UAT Tests
├── pyproject.toml          # Project Config
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE) for details.
