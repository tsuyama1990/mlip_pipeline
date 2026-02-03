# PYACEMAKER

**The Zero-Configuration Autonomous Pipeline for Machine Learning Interatomic Potentials.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

PYACEMAKER democratizes atomistic modeling by automating the complex lifecycle of ACE (Atomic Cluster Expansion) potentials. It orchestrates Structure Generation, DFT Calculations, and Model Training into a self-healing, active-learning loop.

## Overview

PYACEMAKER is an intelligent orchestrator that manages the workflow of creating Machine Learning Interatomic Potentials. It is designed to be:
-   **Zero-Config**: Minimal user setup required.
-   **Modular**: Components (DFT, MD, Training) are swappable via strict interfaces.
-   **Robust**: Type-safe configuration and execution.

## Features

-   **CLI Interface**: Easy-to-use command line interface with `typer`.
-   **Strict Configuration**: Validated `config.yaml` using Pydantic schemas.
-   **Centralized Logging**: Structured logging for all operations.
-   **Mock Mode**: Run the entire pipeline logic without external heavy dependencies (DFT/MD) for testing and development.
-   **Modular Architecture**: Defined Protocols for Explorer, Oracle, Trainer, and Validator.

## Requirements

-   **Python**: 3.12+
-   **Package Manager**: `uv` (Recommended) or `pip`
-   **Operating System**: Linux/macOS

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Install Dependencies**
    Using `uv` for fast dependency resolution:
    ```bash
    uv sync
    ```

## Usage

### Basic Execution

To run the pipeline with a configuration file:

```bash
uv run python -m mlip_autopipec run config.yaml
```

### Example Configuration (`config.yaml`)

```yaml
project_name: "MyProject"
dft:
  code: "qe"
  ecutwfc: 40.0
  kpoints: [2, 2, 2]
training:
  code: "pacemaker"
  cutoff: 5.0
  max_generations: 10
exploration:
  strategy: "random"
  max_temperature: 1000.0
  steps: 100
```

## Architecture/Structure

```text
.
├── config.yaml             # Example configuration
├── src/                    # Source code
│   └── mlip_autopipec/
│       ├── config/         # Pydantic configuration models
│       ├── domain_models/  # Domain entities
│       ├── interfaces/     # Abstract base classes (Protocols)
│       ├── orchestration/  # Core logic and state machine
│       ├── physics/        # Scientific implementations (DFT/MD)
│       ├── utils/          # Utilities (Logging, etc.)
│       └── validation/     # Validation logic
├── tests/                  # Unit and Integration tests
└── dev_documents/          # Design specifications
```

## Roadmap

-   **Cycle 01 (Completed)**: Core Framework, CLI, Config, Mocks.
-   **Cycle 02**: Oracle Implementation (Quantum Espresso integration).
-   **Cycle 03**: Structure Generation (MD/MC strategies).
-   **Cycle 04**: Trainer Implementation (Pacemaker integration).
-   **Cycle 05**: Active Learning Loop (On-the-fly dynamics).
-   **Cycle 06**: Advanced Validation & Deployment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
