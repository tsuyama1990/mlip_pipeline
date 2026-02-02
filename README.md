# PYACEMAKER

**Automated Machine Learning Interatomic Potential Construction System**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**PYACEMAKER** democratizes the creation of state-of-the-art Machine Learning Interatomic Potentials (MLIPs). By leveraging the Atomic Cluster Expansion (ACE) framework and an intelligent Orchestrator, it allows materials scientists to go from "Chemical Composition" to "Production-Ready Potential" with zero coding required.

## Key Features

*   **Zero-Config Workflow**: Define your system in a single `config.yaml` and let the Orchestrator handle the rest.
*   **Robust Architecture**: Pydantic-based strict configuration validation and atomic state persistence ensures your long-running jobs are safe from crashes.
*   **Mock/Dry-Run Mode**: Test your pipeline logic without expensive DFT/MD calculations using the built-in mock mode.
*   **(Planned) Active Learning**: Drastically reduces DFT costs by intelligently selecting structures.
*   **(Planned) Physics-Informed Robustness**: Hybrid potentials (ACE + ZBL/LJ) for stability.
*   **(Planned) Automated Validation**: Rigorous Phonon and Elasticity tests.

## Architecture Overview

PYACEMAKER uses a modular architecture where a central Orchestrator manages the flow of data between specialized agents.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]

    subgraph "Active Learning Loop"
        Orch -->|1. Explore| Gen[Structure Generator]
        Gen -->|2. Sample| Dyn[Dynamics Engine (LAMMPS/EON)]
        Dyn -->|3. Halt on Uncertainty| Select[Selection Logic]
        Select -->|4. Label| Oracle[Oracle (DFT/QE)]
        Oracle -->|5. Train| Trainer[Trainer (Pacemaker)]
        Trainer -->|6. Update| Dyn
    end

    Trainer -->|7. Verify| Val[Validator]
    Val -->|8. Report| Report[HTML Report]
```

## Prerequisites

*   **Python 3.12+**
*   **uv** (recommended for Python dependency management)
*   *(Optional)* **Pacemaker** (`pace_train`)
*   *(Optional)* **LAMMPS** (`lmp`)
*   *(Optional)* **Quantum Espresso** (`pw.x`)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install dependencies using `uv`**:
    ```bash
    uv sync
    # Or install in editable mode
    uv pip install -e .
    ```

## Usage

### Quick Start

1.  **Initialize a new project**:
    Create a working directory and a configuration file.
    ```bash
    mkdir my_project
    cd my_project
    ```
    Create `config.yaml`:
    ```yaml
    project:
      name: "MyProject"
    training:
      dataset_path: "data.pckl"
      max_epochs: 100
    orchestrator:
      max_iterations: 5
    ```

2.  **Run the Orchestrator**:
    ```bash
    uv run python -m mlip_autopipec.main config.yaml
    ```

3.  **Monitor Progress**:
    The system will create `workflow_state.json` and logs to track progress.

## Development

```bash
# Run tests
uv run pytest

# Linting
uv run ruff check .
uv run mypy .
```

## Project Structure

```
mlip-autopipec/
├── src/mlip_autopipec/     # Main package
│   ├── config/             # Pydantic schemas
│   ├── domain_models/      # Core data structures
│   ├── orchestration/      # Logic flow and State
│   └── physics/            # Interface adapters
├── tests/                  # Test suite
└── dev_documents/          # Documentation
```
