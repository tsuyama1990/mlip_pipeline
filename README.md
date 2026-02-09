# PyAceMaker: Automated MLIP Construction System

![Status](https://img.shields.io/badge/status-active_development-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAceMaker** is a fully automated, "Zero-Config" system for constructing and operating Machine Learning Interatomic Potentials (MLIP) using the **Pacemaker** (Atomic Cluster Expansion) engine. It empowers materials scientists to generate "State-of-the-Art" potentials with minimal manual intervention, bridging the gap between high-accuracy DFT and large-scale Molecular Dynamics.

---

## 🚀 Key Features

*   **Zero-Config Workflow**: Initiate a complete training pipeline—from structure generation to validated potential—with a single YAML file.
*   **Active Learning Loop**: autonomous "Halt & Diagnose" mechanism that detects uncertainty during simulations, performs targeted DFT calculations, and retrains the model on-the-fly.
*   **Physics-Informed Robustness**: Enforces core repulsion and uses hybrid potentials (ACE + ZBL/LJ) to prevent physical catastrophes like atomic fusion during high-energy events.
*   **Multi-Scale Dynamics**: Seamlessly integrates standard MD (LAMMPS) with long-timescale Adaptive Kinetic Monte Carlo (EON) to explore diffusive phenomena.
*   **Data Efficiency**: Utilizes D-optimality (Active Set Selection) to achieve high accuracy with 1/10th of the DFT cost compared to random sampling.

## 🏗️ Architecture Overview

The system is orchestrated by a central Python application that manages the data flow between specialized components.

```mermaid
graph TD
    subgraph Control Plane
        Orchestrator[Orchestrator]
        Config[Configuration Manager]
    end

    subgraph Core Modules
        Gen[Structure Generator]
        Oracle[Oracle (DFT)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics Engine (LAMMPS/EON)]
        Val[Validator]
    end

    Config --> Orchestrator
    Orchestrator --> Gen
    Orchestrator --> Oracle
    Orchestrator --> Trainer
    Orchestrator --> Dyn
    Orchestrator --> Val

    Gen -- "Candidate Structures" --> Oracle
    Dyn -- "Halted Structures" --> Gen
    Oracle -- "Labelled Data" --> Trainer
    Trainer -- "New Potential" --> Dyn
```

## 🛠️ Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended for dependency management) or `pip`
*   **External Engines** (Optional for full execution, mocked for CI):
    *   Quantum Espresso (`pw.x`)
    *   LAMMPS (`lmp`) with USER-PACE package
    *   Pacemaker (`pace_train`, `pace_activeset`)
    *   EON (`eonclient`)

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    Using `uv` (faster, recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -e .[dev]
    ```

3.  **Environment Setup**
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    # Edit config.yaml to point to your external binaries if needed
    ```

## ⚡ Usage

### Quick Start
To run the full active learning pipeline with the default configuration:

```bash
uv run python -m mlip_autopipec.main --config config.yaml
```

### Running Tutorials
We provide Jupyter notebooks to demonstrate key features (Mock Mode enabled by default):

```bash
uv run jupyter notebook tutorials/
```

*   `01_MgO_FePt_Training.ipynb`: Train initial potentials.
*   `03_Deposition_MD.ipynb`: Run deposition simulations.

## 💻 Development Workflow

We enforce strict code quality standards using `ruff` and `mypy`.

### Running Tests
```bash
uv run pytest
```

### Linting & Type Checking
```bash
uv run ruff check .
uv run mypy .
```

The system is developed in **8 Sequential Cycles**:
1.  **Core Framework**: Config, Logging, Factory.
2.  **Structure Generator**: Adaptive sampling.
3.  **Oracle**: DFT automation & embedding.
4.  **Trainer**: Pacemaker wrapper & active set.
5.  **Dynamics**: LAMMPS & Hybrid potentials.
6.  **OTF Loop**: The main active learning logic.
7.  **Advanced Dynamics**: EON/kMC integration.
8.  **Validator**: Phonon & Elastic checks.

## 📂 Project Structure

```
.
├── dev_documents/          # System specs & prompt engineering docs
├── src/
│   └── mlip_autopipec/
│       ├── core/           # Orchestrator & Logic
│       ├── components/     # Generator, Oracle, Trainer, etc.
│       └── domain_models/  # Pydantic data models
├── tests/                  # Unit & Integration tests
├── tutorials/              # Jupyter notebooks
├── pyproject.toml          # Project configuration
└── README.md
```

## 📄 License

MIT License. See `LICENSE` for details.
