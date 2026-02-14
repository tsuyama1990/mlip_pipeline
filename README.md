# PYACEMAKER: Automated MLIP Construction System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)

**PYACEMAKER** is an automated system designed to democratize the creation of Machine Learning Interatomic Potentials (MLIP). By leveraging the Atomic Cluster Expansion (ACE) formalism via Pacemaker, it enables materials scientists to generate "State-of-the-Art" potentials with minimal manual intervention.

From a single configuration file, PYACEMAKER orchestrates the entire lifecycle: generating candidate structures, running DFT calculations (Oracle), training the model (Trainer), and validating its physics (Validator). It features a robust "Active Learning" loop that autonomously detects high-uncertainty regions during Molecular Dynamics (MD) simulations and retrains the potential on-the-fly.

## Key Features

-   **Zero-Config Automation**: Define your material system in `config.yaml` and let the system handle structure generation, DFT, training, and validation.
-   **Active Learning Loop**: Uses "Halt & Diagnose" logic to monitor MD simulations. If uncertainty ($\gamma$) spikes, the simulation halts, and the problematic structure is automatically sent for labeling and retraining.
-   **Physics-Informed Robustness**: Enforces Hybrid Potentials (ACE + ZBL/LJ) to prevent unphysical behavior (e.g., core collapse) in unknown regions.
-   **Data Efficiency**: employs D-optimality (Active Set Selection) to achieve DFT-level accuracy with 1/10th the data of random sampling.
-   **Advanced Dynamics**: Seamlessly integrates with EON for Adaptive Kinetic Monte Carlo (aKMC) to explore long-timescale phenomena like diffusion and ripening.

## Architecture Overview

PYACEMAKER follows a modular Orchestrator-Worker pattern.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]

    subgraph "Core System"
        Orch -->|Request Structures| Gen[Structure Generator]
        Orch -->|Submit Jobs| Oracle[Oracle (DFT)]
        Orch -->|Train Model| Trainer[Trainer (Pacemaker)]
        Orch -->|Run Sim| Dynamics[Dynamics Engine]
        Orch -->|Verify| Valid[Validator]
    end

    subgraph "External Tools"
        Gen -.->|M3GNet| M3G[Universal Potential]
        Oracle -.->|QE/VASP| DFT[DFT Code]
        Trainer -.->|PACE| PACE[Pacemaker]
        Dynamics -.->|LAMMPS| LMP[LAMMPS]
        Dynamics -.->|EON| EON[EON Client]
        Valid -.->|Phonopy| Phono[Phonopy]
    end

    subgraph "Data Storage"
        DB[(Dataset .pckl)]
        Pot[(Potential .yace)]
    end

    Oracle -->|Forces/Energy| DB
    Trainer -->|Read| DB
    Trainer -->|Write| Pot
    Dynamics -->|Read| Pot
    Dynamics -->|High Uncertainty| Orch
```

## Prerequisites

-   **Python 3.11+**
-   **LAMMPS** (with USER-PACE package installed)
-   **Pacemaker** (ACE training engine)
-   **Quantum Espresso** (or VASP) for DFT calculations
-   **EON** (for kMC simulations)
-   **uv** (Recommended for dependency management)

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
    *Alternatively, using pip:*
    ```bash
    pip install -e .[dev]
    ```

3.  **Configure environment:**
    Copy the example configuration and adjust paths to your DFT/LAMMPS executables.
    ```bash
    cp config.example.yaml config.yaml
    ```

## Usage

**Run the full active learning loop:**
```bash
uv run pyacemaker run config.yaml
```

**Validate an existing potential:**
```bash
uv run pyacemaker validate potential.yace --config config.yaml
```

**Quick Start (Mock Mode):**
To run a test drive without external heavy codes (DFT/LAMMPS), enable CI mode:
```bash
export CI=true
uv run pytest tests/
```

## Development Workflow

We follow a strictly typed, test-driven development process.

-   **Run Tests:**
    ```bash
    uv run pytest
    ```
-   **Run Linters:**
    ```bash
    uv run ruff check .
    uv run mypy .
    ```
-   **Development Cycles:**
    The project is implemented in 6 cycles. See `dev_documents/system_prompts/CYCLE{xx}/SPEC.md` for details.

## Project Structure

```text
pyacemaker/
├── src/
│   └── pyacemaker/
│       ├── core/       # Configuration, Logging, Base Classes
│       ├── generator/  # Structure Generation (Adaptive Policy)
│       ├── oracle/     # DFT Automation (ASE Interface)
│       ├── trainer/    # Pacemaker Wrapper & Active Set
│       ├── dynamics/   # LAMMPS & EON Interface
│       └── validator/  # Physics Checks (Phonon, EOS)
├── tests/              # Unit & Integration Tests
├── dev_documents/      # Architecture & Specs
├── pyproject.toml      # Project Configuration
└── README.md           # This file
```

## License

MIT License. See `LICENSE` for details.
