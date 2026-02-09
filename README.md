# PYACEMAKER: Automated MLIP Construction System

![Status](https://img.shields.io/badge/Status-Prototype-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

**PYACEMAKER** is a "Zero-Config" automation system for constructing and operating Machine Learning Interatomic Potentials (MLIPs). It democratizes computational materials science by enabling researchers to generate state-of-the-art ACE (Atomic Cluster Expansion) potentials with minimal human intervention, bridging the gap between high-accuracy DFT and large-scale Molecular Dynamics.

## Key Features

*   **Zero-Config Workflow**: Define your material and target accuracy in a single `config.yaml`, and the system handles the rest.
*   **Active Learning with D-Optimality**: Minimizes expensive DFT calculations by selecting only the most information-rich structures for training (Data Efficiency > 10x vs Random).
*   **Physics-Informed Robustness**: Automatically constructs "Hybrid Potentials" (ACE + ZBL/LJ) to prevent unphysical behavior and simulation crashes in high-energy regimes.
*   **Self-Healing Dynamics**: "On-the-Fly" (OTF) monitoring detects when simulations enter unknown territory, halts execution, and triggers an autonomous retraining loop.
*   **Multi-Scale Capability**: Seamlessly integrates Molecular Dynamics (LAMMPS) and Adaptive Kinetic Monte Carlo (EON) to cover time scales from femtoseconds to hours.

## Architecture Overview

PYACEMAKER orchestrates a cycle of Exploration, Labeling, Training, and Validation.

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]

    subgraph "Core Components"
        Orch --> Gen[Structure Generator]
        Orch --> Dyn[Dynamics Engine]
        Orch --> Ora[Oracle (DFT)]
        Orch --> Trn[Trainer (Pacemaker)]
        Orch --> Val[Validator]
    end

    subgraph "External Tools"
        Gen -.-> Pym[Pymatgen]
        Dyn -.-> LAMMPS[LAMMPS]
        Dyn -.-> EON[EON (kMC)]
        Ora -.-> QE[Quantum Espresso]
        Trn -.-> PACE[Pacemaker]
    end

    Gen -- "Candidates" --> Ora
    Dyn -- "Halted Structures" --> Gen
    Ora -- "Labeled Data" --> Trn
    Trn -- "Potential.yace" --> Dyn
```

## Prerequisites

*   **Python**: 3.12 or higher.
*   **uv**: An extremely fast Python package installer and resolver.
*   **External Binaries** (for Real Mode):
    *   `lammps` (with USER-PACE package installed).
    *   `pw.x` (Quantum Espresso) or VASP.
    *   `pace_train` / `pace_activeset` (Pacemaker).

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install dependencies using `uv`**:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Copy the example configuration and adjust paths to your external binaries.
    ```bash
    cp config.example.yaml config.yaml
    ```

## Usage

### Quick Start

To run the full active learning pipeline:

```bash
uv run python -m mlip_autopipec.main --config config.yaml
```

The system will:
1.  Create an `active_learning/` directory.
2.  Generate initial structures (Random/M3GNet).
3.  Run DFT calculations (Oracle).
4.  Train the ACE potential (Trainer).
5.  Run MD validation/exploration (Dynamics).
6.  Repeat until convergence.

### Running Tutorials

We provide Jupyter Notebooks to demonstrate key workflows:

```bash
uv run jupyter notebook tutorials/
```

*   `01_MgO_FePt_Training.ipynb`: Train a potential from scratch.
*   `02_Deposition_and_Ordering.ipynb`: Simulate thin film growth.

## Development Workflow

This project follows a strict cycle-based development plan (Cycle 01 - 08).

### Running Tests

```bash
uv run pytest
```

### Linting & Code Quality

We enforce strict type checking and style guidelines.

```bash
uv run ruff check .
uv run mypy .
```

## Project Structure

```
mlip-pipeline/
├── dev_documents/          # Specs and Design Docs
├── src/
│   └── mlip_autopipec/
│       ├── core/           # Orchestrator & State Logic
│       ├── components/     # Generator, Oracle, Trainer, etc.
│       ├── domain_models/  # Pydantic Schemas
│       └── interfaces/     # External Code Adapters
├── tests/                  # Pytest Suite
└── tutorials/              # Jupyter Notebooks
```

## License

MIT License. See `LICENSE` for details.
