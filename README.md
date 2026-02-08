# PYACEMAKER: Automated MLIP Construction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/mlip-pipeline)

**PYACEMAKER** (Python Atomic Cluster Expansion Maker) is an automated system designed to democratize the creation of high-quality Machine Learning Interatomic Potentials (MLIP). By orchestrating the entire active learning loop—from structure generation and DFT calculation to potential fitting and validation—it allows researchers to develop "State-of-the-Art" potentials with minimal manual intervention.

## Key Features

-   **Zero-Config Workflow**: Define your material system in a single YAML file and let the system handle the rest.
-   **Active Learning with Uncertainty**: Drastically reduces DFT costs by only calculating structures where the model is uncertain.
-   **Robust Architecture**: Built on a modular, type-safe foundation using Pydantic and strict interfaces.
-   **Self-Healing**: Designed to recover from failures automatically (future cycles).
-   **Automated Validation**: Rigorous testing of potentials before deployment.

## Architecture Overview

The system follows a hub-and-spoke architecture where a central Orchestrator manages the data flow between specialized components.

```mermaid
graph TD
    User[User] -->|config.yaml| Orch[Orchestrator]
    Orch -->|Init| Gen[Structure Generator]
    Orch -->|Loop| Dyn[Dynamics Engine]

    subgraph Active Learning Loop
        Gen -->|Candidate Structures| Oracle[Oracle (DFT)]
        Dyn -->|High Uncertainty Structures| Oracle
        Oracle -->|Labeled Data| DB[(Dataset)]
        DB -->|Training Set| Trainer[Trainer (Pacemaker)]
        Trainer -->|potential.yace| Val[Validator]
        Val -- Pass --> Orch
        Val -- Fail --> Gen
    end
```

## Prerequisites

-   **Python**: Version 3.12 or higher.
-   **Package Manager**: `uv` (recommended) or `pip`.

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install Dependencies (using uv)**
    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Sync dependencies
    uv sync
    ```

3.  **Activate Virtual Environment**
    ```bash
    source .venv/bin/activate
    ```

## Usage

To start a new project, create a configuration file (e.g., `config.yaml`) and run the pipeline.

### Quick Start

1.  **Create a Config File**
    ```yaml
    workdir: "runs/demo_run"
    max_cycles: 5
    generator:
      type: "mock"
    oracle:
      type: "mock"
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    validator:
      type: "mock"
    ```

2.  **Run the Pipeline**
    ```bash
    mlip-pipeline run config.yaml
    ```

## Development Workflow

We follow the AC-CDD (Architectural-Core Cycle Driven Development) methodology.

-   **Running Tests**:
    ```bash
    PYTHONPATH=src pytest
    ```

-   **Linting & Formatting**:
    ```bash
    ruff check .
    ruff format .
    ```

-   **Type Checking**:
    ```bash
    mypy .
    ```

## Project Structure

```ascii
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── components/   # Core modules (Generator, Oracle, etc.)
│       ├── core/         # Orchestrator & Dataset logic
│       ├── domain_models/# Pydantic schemas
│       ├── interfaces/   # Abstract Base Classes
│       └── main.py       # CLI Entry point
├── tests/                # Unit & Integration tests
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
