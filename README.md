# PYACEMAKER: Autopilot for Machine Learning Potentials

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an autonomous system that democratizes the creation of State-of-the-Art Machine Learning Interatomic Potentials (MLIPs). It orchestrates the entire lifecycle—from structure generation and DFT labeling to active learning and validation—allowing researchers to generate robust potentials for complex materials with a single configuration file.

---

## 🚀 Key Features

-   **Zero-Config Workflow**: Define your intent (e.g., "Fe-Pt Alloy") in a simple YAML file, and the system handles the rest. No complex Python scripting required.
-   **Data Efficiency**: Utilizes **Active Learning** with D-Optimality (MaxVol) selection to achieve DFT-level accuracy with 1/10th of the training data compared to random sampling.
-   **Physics-Informed Robustness**: Enforces a "Delta Learning" architecture (ACE + ZBL/LJ baseline) to ensure physical correctness and stability even in far-from-equilibrium regimes.
-   **Self-Healing Oracle**: Automatically detects and fixes DFT convergence errors (e.g., mixing beta adjustment) without human intervention.
-   **Scalable Architecture**: Modular design supports seamless transition from local workstations to HPC clusters, integrating MD (LAMMPS) and kMC (EON) for multi-scale simulation.

---

## 🏗 Architecture Overview

The system operates as a centralized orchestrator managing specialized micro-components.

```mermaid
graph TD
    subgraph "Control Plane"
        Orch[Orchestrator]
        Config[Global Config]
    end

    subgraph "Compute Plane"
        SG[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        DE[Dynamics Engine (MD/kMC)]
        Val[Validator]
    end

    Config --> Orch
    Orch --> SG
    Orch --> Oracle
    Orch --> Trainer
    Orch --> DE
    Orch --> Val

    DE -- "Uncertainty Halt" --> SG
    SG -- "Candidates" --> Oracle
    Oracle -- "Labeled Data" --> Trainer
    Trainer -- "New Potential" --> DE
```

---

## 🛠 Prerequisites

-   **Python 3.12+**
-   **uv** (Fast Python package manager)
-   **Quantum Espresso** (`pw.x`) for DFT calculations (Optional for Mock Mode)
-   **LAMMPS** with `USER-PACE` package for MD simulations
-   **Pacemaker** for training ACE potentials

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install Dependencies**
    We use `uv` for lightning-fast dependency management.
    ```bash
    uv sync
    ```

3.  **Activate Environment**
    ```bash
    source .venv/bin/activate
    ```

4.  **Initialize Configuration**
    Generate a default configuration file.
    ```bash
    pyacemaker init
    ```

## 🏃 Usage

### Quick Start (Mock Mode)
To verify the installation without running heavy calculations, use the Mock Mode. This runs the full pipeline logic using fake data.

1.  Edit `config.yaml` and set all component types to `mock`.
2.  Run the pipeline:
    ```bash
    pyacemaker run --config config.yaml
    ```

### Production Run
To train a real potential for a Silicon crystal:

1.  Configure `oracle` to use `qe` and provide pseudopotentials.
2.  Run the command:
    ```bash
    pyacemaker run --config config.yaml
    ```

### Other Commands
-   **Compute DFT**: `pyacemaker compute --structure struct.xyz`
-   **Train Potential**: `pyacemaker train --dataset data.pckl.gzip`
-   **Validate**: `pyacemaker validate --potential potential.yace`

## 💻 Development Workflow

We follow the **AC-CDD** (Architecturally Constrained Cycle-Driven Development) methodology.

### Running Tests
Execute the full test suite (Unit + Integration):
```bash
uv run pytest
```

### Linting & Formatting
Enforce code quality standards:
```bash
uv run ruff check .
uv run mypy .
```

### Directory Structure
```
mlip-pipeline/
├── config.yaml               # User configuration
├── src/
│   └── mlip_autopipec/       # Source code
│       ├── domain_models/    # Pydantic data models
│       ├── interfaces/       # Abstract Base Classes
│       ├── infrastructure/   # Concrete implementations (QE, LAMMPS, etc.)
│       └── orchestrator/     # Main control logic
├── tests/                    # Test suite
├── dev_documents/            # Architecture & Cycle specs
└── tutorials/                # Jupyter Notebooks for UAT
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
