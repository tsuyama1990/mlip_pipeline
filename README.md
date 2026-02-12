# PYACEMAKER: Automated MLIP Construction & Operation System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an end-to-end automated system for constructing "State-of-the-Art" Machine Learning Interatomic Potentials (MLIP) using the Atomic Cluster Expansion (ACE) formalism. It empowers materials scientists to go from **Zero to Simulation** without writing code, automating the complex "Explore-Label-Train" active learning cycle.

## 🚀 Key Features

*   **Zero-Config Workflow**: Define your material and goals in a single `config.yaml`. The system handles the rest.
*   **Data Efficiency**: Uses **D-Optimality (Active Learning)** to select only the most informative structures for DFT, reducing computational cost by 90% compared to random sampling.
*   **Physics-Informed Robustness**: Automatically blends ACE with ZBL/LJ baselines ("Hybrid Potential") and monitors uncertainty ($\gamma$) to prevent unphysical crashes during MD.
*   **Time-Scale Extension**: Seamlessly bridges nanosecond MD simulations with second-scale **Adaptive Kinetic Monte Carlo (aKMC)** via EON integration.

## 🏗️ Architecture Overview

The system follows a modular Hub-and-Spoke architecture orchestrated by a central Python controller.

```mermaid
graph TD
    User[User] --> Config[config.yaml]
    Config --> Orch[Orchestrator]

    Orch --> Gen[Structure Generator]
    Orch --> Oracle[Oracle (DFT)]
    Orch --> Train[Trainer (Pacemaker)]
    Orch --> Dyn[Dynamics Engine]

    Dyn --> LAMMPS[LAMMPS (MD)]
    Dyn --> EON[EON (kMC)]
    Oracle --> QE[Quantum Espresso]
    Train --> PACE[Pacemaker]

    Dyn -- "Halt (High Uncertainty)" --> Orch
    Orch -- "New Candidates" --> Oracle
    Oracle -- "Labeled Data" --> Train
    Train -- "New Potential" --> Dyn
```

## 🛠️ Prerequisites

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Engines** (Optional for Mock Mode):
    *   LAMMPS (with USER-PACE package)
    *   Quantum Espresso (`pw.x`)
    *   Pacemaker (`pace_train`)
    *   EON (`eonclient`)

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    We recommend using `uv` for fast, reproducible environments.
    ```bash
    uv sync --dev
    ```

3.  **Initialize Configuration**
    Generate a default configuration file.
    ```bash
    uv run mlip-runner init
    ```

## 🚀 Usage

### Quick Start
To run the full Active Learning pipeline:

```bash
uv run mlip-runner run config.yaml
```

### Tutorials
Check the `tutorials/` directory for Jupyter Notebooks demonstrating real-world scenarios:
*   `01_MgO_FePt_Training.ipynb`: Train potentials for MgO and FePt.
*   `02_Deposition_and_Ordering.ipynb`: Run Hybrid MD deposition and aKMC ordering.

## 💻 Development Workflow

We enforce strict code quality standards.

**Run Tests:**
```bash
uv run pytest
```

**Run Linters:**
```bash
uv run ruff check .
uv run mypy .
```

## 📂 Project Structure

```ascii
src/mlip_autopipec/
├── core/               # Config, Logging, State
├── domain_models/      # Pydantic Schemas
├── structure_generator/# MD/MC/M3GNet Generators
├── oracle/             # DFT Interface (QE/VASP)
├── trainer/            # Pacemaker Interface
├── dynamics/           # LAMMPS & EON Drivers
├── orchestrator/       # Main Loop Logic
└── validator/          # Phonon & Elastic Checks
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
