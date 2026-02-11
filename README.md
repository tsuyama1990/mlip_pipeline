# PyAceMaker

**Democratising Machine Learning Interatomic Potentials (MLIP) with Zero-Config Active Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mlip-pipeline/pyacemaker/ci.yml)](https://github.com/mlip-pipeline/pyacemaker/actions)

## 🚀 Overview

**PyAceMaker** is an automated pipeline designed to bridge the gap between complex atomic simulations and user-friendly potential generation. Built on top of the powerful **Pacemaker** (Atomic Cluster Expansion) engine, it enables researchers to create "State-of-the-Art" machine learning potentials with a single configuration file.

Traditionally, constructing an MLIP required deep expertise in DFT, MD, and fitting algorithms. PyAceMaker automates the entire loop: from adaptive structure generation and DFT labeling (Oracle) to Active Learning training and final validation.

## ✨ Key Features

*   **Zero-Config Workflow**: Define your material and goals in `config.yaml`, and let the Orchestrator handle the rest.
*   **Active Learning**: Drastically reduces DFT costs by selecting only the most informative structures using D-Optimality (Active Set Selection).
*   **Physics-Informed Robustness**: Automatically enforces a physical baseline (Lennard-Jones/ZBL) to prevent non-physical behavior and crashes in extrapolation regions.
*   **On-the-Fly (OTF) Self-Healing**: Monitors simulation uncertainty in real-time. If the potential becomes unreliable, the system halts, learns from the failure, retrains, and resumes automatically.
*   **Scalable & Extensible**: Designed to scale from local testing loops to massive HPC simulations involving Deposition and Kinetic Monte Carlo (aKMC).

## 🏗️ Architecture

PyAceMaker follows a modular "Hub-and-Spoke" architecture orchestrated by a central controller.

```mermaid
graph TD
    User[User] -->|Config (yaml)| Orch[Orchestrator]
    Orch -->|Manage| State[State Manager]

    subgraph "Active Learning Loop"
        Orch -->|1. Explore| Gen[Structure Generator]
        Orch -->|2. Simulate & Halt| Dyn[Dynamics Engine]
        Orch -->|3. Label| Oracle[Oracle (DFT)]
        Orch -->|4. Train| Trainer[Trainer (Pacemaker)]
        Orch -->|5. Verify| Valid[Validator]
    end

    Gen -->|Candidates| Oracle
    Dyn -->|Halt Structures| Gen
    Oracle -->|Refined Data| Trainer
    Trainer -->|Potential (yace)| Dyn
    Trainer -->|Potential (yace)| Valid
```

## 🛠️ Prerequisites

*   **Python 3.12+**
*   **uv** (Recommended package manager)
*   **Docker** (Optional, for containerized execution)
*   **External Engines** (for full production runs):
    *   **Pacemaker**: `pace_train`, `pace_collect`
    *   **LAMMPS**: With `USER-PACE` package
    *   **Quantum Espresso** (or VASP): `pw.x`

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mlip-pipeline/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install dependencies with uv**:
    ```bash
    uv sync
    ```

3.  **Activate the environment**:
    ```bash
    source .venv/bin/activate
    ```

## 🚦 Usage

### Quick Start (Mock Mode)
To test the pipeline without external physics engines, run the built-in mock loop:

1.  **Initialize a new project**:
    ```bash
    pyacemaker init my_project
    cd my_project
    ```

2.  **Run the Active Learning Loop**:
    ```bash
    pyacemaker run-loop --config config.yaml
    ```
    *This will execute a simulation using Mock components, demonstrating the workflow logic.*

### Production Run
Edit `config.yaml` to specify your DFT and LAMMPS settings, then run the same command.

```yaml
orchestrator:
  mode: real
  n_iterations: 10

oracle:
  type: espresso
  command: "mpirun -np 32 pw.x"
```

## 💻 Development Workflow

We follow a strictly cycled development process (Cycles 01-08).

### Running Tests
```bash
uv run pytest
```

### Linting & Formatting
We enforce strict code quality using `ruff` and `mypy`.
```bash
uv run ruff check .
uv run mypy .
```

## 📂 Project Structure

```
.
├── dev_documents/          # Specs and Architecture docs
├── src/
│   └── mlip_autopipec/     # Source code
│       ├── components/     # Logic modules (Oracle, Trainer, etc.)
│       ├── core/           # Orchestrator & Utils
│       └── domain_models/  # Pydantic Schemas
├── tests/                  # Test suite
├── pyproject.toml          # Config & Dependencies
└── README.md
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
