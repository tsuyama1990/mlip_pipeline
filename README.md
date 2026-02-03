# PYACEMAKER

**The Zero-Configuration Autonomous Pipeline for Machine Learning Interatomic Potentials.**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/example/pyacemaker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

PYACEMAKER democratizes atomistic modeling by automating the complex lifecycle of ACE (Atomic Cluster Expansion) potentials. It orchestrates Structure Generation, DFT Calculations (Quantum Espresso), and Model Training (Pacemaker) into a self-healing, active-learning loop that delivers "State-of-the-Art" accuracy with minimal human intervention.

## Key Features

-   **Zero-Config Workflow**: Define your material in a single `config.yaml` and let the system handle the rest.
-   **Active Learning**: Automatically identifies and explores "uncertain" regions of the chemical space using Extrapolation Grade ($\gamma$) monitoring.
-   **Physics-Informed Robustness**: Enforces physical baselines (Lennard-Jones/ZBL) to ensure stability in high-energy regimes where data is scarce.
-   **Self-Healing Oracle**: Automatically recovers from DFT convergence failures by adjusting mixing parameters and smearing settings.
-   **Timescale Bridging**: Seamlessly integrates with Adaptive Kinetic Monte Carlo (EON) to explore rare events beyond the reach of standard MD.

## Architecture Overview

PYACEMAKER follows a Hub-and-Spoke architecture centered around an intelligent Orchestrator.

```mermaid
graph TD
    subgraph User Space
        Config[config.yaml] --> Orch[Orchestrator]
    end

    subgraph Core System
        Orch -->|1. Request Structures| Explorer[Structure Generator]
        Explorer -->|2. Candidate Structures| Orch

        Orch -->|3. Filter & Request Energy| Oracle[Oracle (DFT)]
        Oracle -->|4. Labelled Data (E, F, V)| Orch

        Orch -->|5. Update Dataset| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. New Potential (.yace)| Orch

        Orch -->|7. Run Validation| Validator[Validator]
        Validator -->|8. Report & Pass/Fail| Orch
    end

    subgraph External Engines
        Explorer -.-> LAMMPS[LAMMPS (MD)]
        Oracle -.-> QE[Quantum Espresso]
        Trainer -.-> PACE[Pacemaker]
    end
```

## Prerequisites

-   **Python**: 3.12+
-   **Package Manager**: `uv` (Recommended) or `pip`
-   **External Tools** (For "Real Mode"):
    -   Quantum Espresso (`pw.x`)
    -   LAMMPS (`lmp_serial` or `lmp_mpi`) with USER-PACE package
    -   Pacemaker (TensorFlow-based)
-   **Docker** (Optional, for containerized execution)

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    Using `uv` for fast dependency resolution:
    ```bash
    uv sync
    ```

3.  **Environment Setup**
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    # Edit config.yaml with your specific settings (pseudopotential paths, etc.)
    ```

## Usage

### Quick Start
To run the full pipeline in default mode:

```bash
uv run python -m mlip_autopipec run config.yaml
```

### Running Tutorials
We provide a set of Jupyter Notebooks to guide you from basic usage to advanced scenarios.

```bash
uv run jupyter notebook tutorials/
```

-   **`01_quickstart_silicon.ipynb`**: Train a simple Silicon potential in under 5 minutes (Mock Mode available).
-   **`04_grand_challenge_fept.ipynb`**: Full workflow for Fe/Pt deposition on MgO.

## Development Workflow

We enforce strict code quality standards.

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
```bash
uv run ruff check .
uv run mypy .
```

### Cycle-Based Development
This project follows a cycle-based implementation plan. Check `dev_documents/system_prompts/` for detailed specifications of each development cycle (CYCLE01 to CYCLE06).

## Project Structure

```text
.
├── config.yaml             # Main configuration
├── src/                    # Source code
│   └── mlip_autopipec/
│       ├── orchestration/  # Workflow logic
│       ├── physics/        # Scientific modules (DFT, MD, ML)
│       └── domain_models/  # Pydantic data models
├── tests/                  # Unit and Integration tests
├── tutorials/              # User guides (Jupyter Notebooks)
└── dev_documents/          # Architecture and Design specs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
