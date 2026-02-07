# PYACEMAKER (MLIP Pipeline)

![Status](https://img.shields.io/badge/Status-Cycle_01-blue)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Democratizing Atomistic Simulations: From Zero to State-of-the-Art in One Config.**

PYACEMAKER is an automated pipeline for constructing and operating high-efficiency Machine Learning Interatomic Potentials (MLIPs).

## Features

*   **Zero-Config Automation**: Define your material system in `config.yaml` and let the orchestrator handle the rest.
*   **Modular Architecture**: Interface-driven design allowing easy swapping of components (Oracle, Trainer, Dynamics).
*   **Mock Components**: Built-in mock implementations for rapid prototyping and testing of the orchestration logic without external dependencies.
*   **Robust Configuration**: Strict validation using Pydantic ensures configuration integrity before execution.

## Requirements

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage

**Run the pipeline:**
```bash
uv run mlip-pipeline run config.yaml
```

**Example Configuration (Mock Mode):**
Create a `config.yaml` file:
```yaml
workdir: "experiments/test_01"
oracle:
  type: "mock"
  noise_level: 0.1
trainer:
  type: "mock"
dynamics:
  type: "mock"
```

Then run:
```bash
uv run mlip-pipeline run config.yaml
```

## Architecture/Structure

```ascii
src/
└── mlip_autopipec/
    ├── domain_models/      # Pydantic data models (Structure, Config, etc.)
    ├── interfaces/         # Abstract Base Classes (Oracle, Trainer, etc.)
    ├── infrastructure/     # Concrete Implementations (Mocks)
    ├── factory.py          # Component Factory
    ├── main.py             # CLI Entry Point
    └── utils/              # Utilities (Logging)
```

## Roadmap

*   **Cycle 02**: Data Generation & Management
*   **Cycle 03**: Oracle Implementation (Quantum Espresso)
*   **Cycle 04**: Trainer Implementation (Pacemaker)
*   **Cycle 05**: Dynamics Implementation (LAMMPS)
*   **Cycle 06**: Full Orchestration & Validation
