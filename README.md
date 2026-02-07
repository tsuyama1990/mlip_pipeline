# PYACEMAKER: Automated Machine Learning Interatomic Potential Pipeline

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-org/mlip-pipeline)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**PYACEMAKER** democratises materials science by enabling researchers to build State-of-the-Art Machine Learning Interatomic Potentials (MLIPs) with zero coding. It automates the complex loop of structure generation, DFT labeling (Quantum Espresso), training (Pacemaker), and validation, allowing you to simulate "hard" problems like hetero-epitaxial growth and phase transitions with DFT accuracy at MD speeds.

---

## 🚀 Key Features

*   **Zero-Config Workflow**: Define your material system (e.g., "FePt on MgO") in a single YAML file and let the Orchestrator handle the rest.
*   **Mock Orchestration**: Complete pipeline flow verification using mock components (Cycle 01).
*   **Active Learning with Self-Healing**: Automatically detects high-uncertainty regions during MD and triggers DFT calculations. The system "heals" itself from DFT convergence failures without human intervention.
*   **Physics-Informed Robustness**: Enforces physical baselines (Lennard-Jones/ZBL) to prevent simulation crashes due to non-physical atomic overlaps in unknown regions.
*   **Time-Scale Bridging**: Seamlessly integrates Molecular Dynamics (for fast processes) and Adaptive Kinetic Monte Carlo (for slow diffusion/ordering), solving the "time-scale problem".
*   **Data Efficiency**: Uses D-Optimality (MaxVol) selection to achieve target accuracy with 10x fewer DFT calculations than random sampling.

---

## 🏗 Architecture Overview

The system follows a hub-and-spoke architecture centered around a Python Orchestrator.

```mermaid
graph TD
    subgraph "Core System"
        Orch[Orchestrator]
        Config[Global Config]
        DB[(Dataset)]
    end

    subgraph "Agents"
        Gen[Structure Generator]
        Oracle[Oracle (DFT/QE)]
        Trainer[Trainer (Pacemaker)]
        Dyn[Dynamics (LAMMPS/EON)]
        Val[Validator]
    end

    User[User] -->|config.yaml| Config
    Orch -->|Read| Config
    Orch -->|Request| Gen
    Gen -->|Structures| Orch
    Orch -->|Submit| Oracle
    Oracle -->|Labels| DB
    DB -->|Train| Trainer
    Trainer -->|Potential| Orch
    Orch -->|Deploy| Dyn
    Dyn -->|Uncertainty| Orch
    Orch -->|Validate| Val
    Val -->|Report| User
```

---

## 🛠 Prerequisites

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **External Tools** (for production runs):
    *   [Quantum Espresso](https://www.quantum-espresso.org/) (`pw.x`)
    *   [LAMMPS](https://www.lammps.org/) (with USER-PACE package)
    *   [Pacemaker](https://pacemaker.readthedocs.io/)
    *   [EON](https://theory.cm.utexas.edu/eon/) (for aKMC)

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install dependencies using uv**:
    ```bash
    uv sync
    ```
    *Or using pip:*
    ```bash
    pip install .
    ```

3.  **Set up environment**:
    Copy the example environment file and configure paths to your external binaries if necessary.
    ```bash
    cp .env.example .env
    ```

---

## 🏃 Usage

### Quick Start
To run a test cycle using **Mock Components** (no external binaries needed):

1.  Create a config file `config.yaml`:
    ```yaml
    workdir: ./output
    max_cycles: 2
    generator: {type: mock}
    oracle: {type: mock}
    trainer: {type: mock}
    dynamics: {type: mock}
    ```

2.  Run the pipeline:
    ```bash
    uv run mlip-pipeline run config.yaml
    ```

### Production Run
To run a real Active Learning cycle for FePt:

```bash
uv run mlip-pipeline run examples/fept_mgo.yaml
```

---

## 💻 Development Workflow

We follow the **AC-CDD** (Architect-Coder-Checker-Developer) methodology with 6 implementation cycles.

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
We enforce strict code quality.
```bash
uv run ruff check .
uv run mypy .
```

### Project Structure
```ascii
src/mlip_autopipec/
├── domain_models/     # Pydantic data models
├── interfaces/        # Abstract Base Classes
├── components/        # Concrete implementations (QE, LAMMPS, etc.)
├── core/              # Orchestrator logic
└── main.py            # CLI entry point

dev_documents/
├── system_prompts/    # Cycle specifications
├── ALL_SPEC.md        # Full requirements
└── FINAL_UAT.md       # User Acceptance Testing plan
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
