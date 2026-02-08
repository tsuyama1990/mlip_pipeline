# PYACEMAKER

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an automated, zero-configuration system for constructing and operating "State-of-the-Art" Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (Atomic Cluster Expansion) engine. It democratises high-accuracy atomic simulations by handling the complex cycle of exploration, DFT calculation, and training autonomously.

## Key Features

1.  **Zero-Config Workflow**: Define your material and goals in a single YAML file, and the system handles the rest—from initial structure generation to final validation.
2.  **Data Efficiency**: Uses Active Learning and Physics-Informed sampling (e.g., adaptive temperature ramping, defect engineering) to achieve high accuracy with 1/10th the DFT cost of random sampling.
3.  **Physics-Informed Robustness**: Enforces "Delta Learning" against a physical baseline (Lennard-Jones/ZBL) to ensure simulations never crash due to non-physical forces in extrapolation regions.
4.  **On-the-Fly (OTF) Learning**: Automatically detects when a simulation enters an unknown region (high uncertainty), halts, performs a DFT calculation on the fly, retrains, and resumes.

## Architecture Overview

PYACEMAKER uses a modular architecture orchestrated by a central Python controller.

```mermaid
graph TD
    User[User] -->|Config (yaml)| Orch[Orchestrator]
    Orch -->|Control| Gen[Structure Generator]
    Orch -->|Control| Dyn[Dynamics Engine]
    Orch -->|Control| Oracle[Oracle (DFT)]
    Orch -->|Control| Trainer[Trainer (Pacemaker)]
    Orch -->|Control| Valid[Validator]

    Gen -->|Candidate Structures| Dyn
    Dyn -->|Exploration (MD/kMC)| OTF{Uncertainty Check}
    OTF -->|High Uncertainty| Oracle
    OTF -->|Low Uncertainty| Dyn
    Oracle -->|Labeled Data| Dataset[(Dataset)]
    Dataset --> Trainer
    Trainer -->|Potential (yace)| Dyn
    Trainer -->|Potential (yace)| Valid
    Valid -->|Report| User
```

## Prerequisites

*   **Python 3.12+**
*   **uv** (Modern Python package manager)
*   **External Engines** (Optional for Mock/Dev mode, required for Production):
    *   **Quantum Espresso** (`pw.x`) for DFT.
    *   **LAMMPS** (`lmp`) for MD.
    *   **Pacemaker** (`pace_train`) for training.

## Installation & Setup

We use `uv` for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/your-org/pyacemaker.git
cd pyacemaker

# Install dependencies
uv sync --dev

# Activate the environment
source .venv/bin/activate
```

## Usage

### Quick Start (Mock Mode)

To verify the installation and see the pipeline in action without heavy computations:

```bash
# Run the pipeline with the example configuration
mlip-pipeline run examples/config_quickstart.yaml
```

### Production Run

1.  Create your configuration file `config.yaml`:
    ```yaml
    workdir: ./my_project
    max_cycles: 10
    generator:
      elements: ["Fe", "Pt"]
    oracle:
      calculator_type: qe
      pseudopotential_dir: /path/to/sssp
    ```
2.  Run the pipeline:
    ```bash
    mlip-pipeline run config.yaml
    ```

## Development Workflow

This project follows the **AC-CDD (Architect-Coder-Cycle-Driven-Development)** methodology. Development is divided into 6 cycles:

1.  **Cycle 01**: Core Framework & Mocks
2.  **Cycle 02**: Data Management & Structure Generator
3.  **Cycle 03**: Oracle (DFT Interface)
4.  **Cycle 04**: Trainer (Pacemaker Integration)
5.  **Cycle 05**: Dynamics Engine (LAMMPS/OTF)
6.  **Cycle 06**: Validation & Full Orchestration

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run linters
ruff check .
mypy .
```

## Project Structure

```ascii
src/
├── mlip_autopipec/
│   ├── main.py                 # CLI Entry Point
│   ├── core/                   # Orchestrator & State
│   ├── components/             # Generator, Oracle, Trainer, Dynamics, Validator
│   ├── domain_models/          # Pydantic Models
│   └── interfaces/             # Abstract Base Classes
dev_documents/
├── system_prompts/
│   ├── SYSTEM_ARCHITECTURE.md  # Detailed Design
│   └── CYCLE{01-06}/           # Cycle-specific specs
└── FINAL_UAT.md                # User Acceptance Test Plan
tutorials/                      # Jupyter Notebooks for Users
```

## License

MIT License. See `LICENSE` for details.
