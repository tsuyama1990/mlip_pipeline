# PyAceMaker (MLIP Pipeline)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PyAceMaker** is an automated "Zero-Config" system for constructing state-of-the-art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) engine. It democratises materials simulation by enabling researchers—even those without deep expertise in data science—to generate robust, production-ready potentials from a single configuration file.

---

## 🚀 Key Features

*   **Zero-Config Workflow**: Automates the entire pipeline from structure generation to DFT calculation, training, and validation.
*   **Active Learning**: Drastically reduces DFT costs (>90%) by intelligently sampling only the most informative structures using D-Optimality and uncertainty quantification.
*   **Hybrid Potentials**: Ensures physical robustness by overlaying a physics-based baseline (ZBL/LJ) on the MLIP, preventing unphysical atomic overlaps and simulation crashes.
*   **Self-Healing Oracle**: Automatically detects and recovers from DFT convergence failures (Quantum Espresso/VASP).
*   **Multi-Scale Dynamics**: Seamlessly bridges Molecular Dynamics (LAMMPS) and Kinetic Monte Carlo (EON) to explore both fast vibrations and slow diffusion/reaction events.

---

## 🏗️ Architecture Overview

The system is built on a modular "Orchestrator-Worker" architecture.

```mermaid
graph TD
    User[User / Config] --> Orch[Orchestrator]
    Orch -->|Request Structures| Gen[Structure Generator]
    Orch -->|Run Simulation| Dyn[Dynamics Engine\n(LAMMPS / EON)]
    Dyn -->|Halt on High Uncertainty| Orch
    Orch -->|Select Candidates| Trainer[Trainer\n(Pacemaker)]
    Trainer -->|Filter (Active Set)| Oracle[Oracle\n(DFT: QE/VASP)]
    Oracle -->|Ground Truth| Orch
    Orch -->|Update Dataset| Trainer
    Trainer -->|Train Potential| Orch
    Orch -->|Validate| Val[Validator]
    Val -->|Pass| Prod[Production Potential]
```

## 📋 Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended for dependency management) or pip
*   **Docker** (Optional, for containerized execution)
*   **External Binaries** (Must be in PATH for full functionality):
    *   `pw.x` (Quantum Espresso)
    *   `lmp_serial` / `lmp_mpi` (LAMMPS)
    *   `pace_train` / `pace_collect` (Pacemaker)
    *   `eonclient` (EON)

## 🛠️ Installation & Setup

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
    Or utilizing `pip`:
    ```bash
    pip install -e .[dev]
    ```

3.  **Configuration**
    Copy the example configuration:
    ```bash
    cp config.example.yaml config.yaml
    # Edit config.yaml to set your material system (e.g., composition: "MgO")
    ```

## 💻 Usage

### Quick Start
To run the full pipeline with default settings:

```bash
uv run python src/mlip_autopipec/main.py --config config.yaml
```

### Tutorials
Check the `tutorials/` directory for Jupyter notebooks demonstrating key workflows:
*   `01_MgO_FePt_Training.ipynb`: Basic training of bulk potentials.
*   `02_Deposition_and_Ordering.ipynb`: Simulating deposition and ordering phenomena.

## 🧑‍💻 Development Workflow

We follow a strict development cycle managed by `uv` and `ruff`.

**Run Tests:**
```bash
uv run pytest
```

**Run Linters:**
```bash
uv run ruff check .
uv run mypy .
```

**Project Cycles:**
Development is broken down into 8 sequential cycles:
1.  **Core Framework**: Orchestrator & Config
2.  **Structure Generator**: Adaptive Sampling
3.  **Oracle**: DFT Automation
4.  **Trainer**: Pacemaker Integration
5.  **Dynamics**: LAMMPS Integration
6.  **OTF Loop**: Active Learning Cycle
7.  **Advanced Dynamics**: EON/kMC
8.  **Validator**: Physics Tests

## 📂 Project Structure

```ascii
src/
├── mlip_autopipec/
│   ├── core/           # Orchestrator & Config Logic
│   ├── components/     # Generator, Oracle, Trainer, Dynamics, Validator
│   ├── domain_models/  # Pydantic Schemas
│   └── utils/          # Logging & Helpers
dev_documents/          # Detailed Specs & UAT Plans
tests/                  # Unit & Integration Tests
tutorials/              # User Guides (Jupyter Notebooks)
```

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
