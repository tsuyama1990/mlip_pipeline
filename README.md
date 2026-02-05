# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**PYACEMAKER** is an automated system for constructing robust Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) formalism. It democratizes access to state-of-the-art potential generation by providing a "Zero-Config" workflow that autonomously iterates through structure generation, DFT calculation, training, and validation.

## Overview

### What is this?
A comprehensive pipeline that automates the generation of machine learning interatomic potentials (MLIPs). It orchestrates the entire lifecycle: generating atomic structures, calculating their properties using DFT, training ACE potentials, and validating them.

### Why?
Manually curating datasets and training potentials is error-prone and requires expert knowledge. PYACEMAKER automates this process, ensuring physics-informed robustness (via ZBL/LJ baselines) and data efficiency through active learning.

## Features

-   **Zero-Config Automation**: Go from chemical composition to a fully trained `.yace` potential with a single YAML file.
-   **Active Learning Loop**: Fully automated cycle of Exploration -> Labeling -> Training -> Validation.
-   **Mock Mode**: Built-in mock components for testing the workflow logic without needing heavy physics engines installed.
-   **Strict Schema Validation**: All inputs and outputs are validated using strict Pydantic models.
-   **Physics-Informed Robustness**: (Planned) Automatically incorporates Lennard-Jones/ZBL baselines.
-   **Self-Healing Oracle**: (Planned) Automated DFT calculations with error recovery.

## Requirements

-   **Python 3.12+**
-   **uv** (recommended for dependency management) or pip
-   **External Physics Codes** (Required only for "Production" mode, not "Mock" mode):
    -   LAMMPS (with USER-PACE package)
    -   Quantum Espresso (pw.x)
    -   Pacemaker

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-pipeline.git
    cd mlip-pipeline
    ```

2.  **Initialize environment**:
    ```bash
    uv sync
    ```

## Usage

The system is controlled via a Command Line Interface (CLI).

### Basic Command

```bash
uv run mlip-pipeline run config.yaml
```

### Configuration Example

Create a file named `config.yaml`:

```yaml
execution_mode: mock  # Use 'mock' to test the loop without physics engines
max_cycles: 5

exploration:
  strategy_name: random
  max_structures: 10

dft:
  calculator: espresso
  encut: 600.0
  kpoints: [2, 2, 2]

training:
  fitting_code: pacemaker
  max_epochs: 100
```

### Mock Mode vs. Real Mode

-   **Mock Mode** (`execution_mode: mock`): Uses internal dummy classes (`MockExplorer`, `MockOracle`, etc.). It generates placeholder XYZ files and random numbers. Useful for CI/CD and verifying workflow logic.
-   **Real Mode** (`execution_mode: production`): (Not fully implemented in Cycle 01) Will invoke actual external binaries (QE, Pacemaker) to perform real physics calculations.

## Architecture/Structure

```ascii
src/mlip_autopipec/
├── config/                  # Pydantic Configuration Models
├── domain_models/           # Domain Entities (Structures, ValidationResult)
├── interfaces/              # Protocol Definitions (Explorer, Oracle, etc.)
├── orchestration/           # Main Loop Logic & Mocks
├── utils/                   # Shared Utilities (Logging)
└── main.py                  # CLI Entrypoint
```

## Development Workflow

We follow a strict quality assurance process.

1.  **Run Tests**:
    ```bash
    # Run all tests
    uv run pytest

    # Run specific test suites
    uv run pytest tests/unit/
    uv run pytest tests/uat/
    ```

2.  **Linting & Type Checking**:
    ```bash
    uv run ruff check .
    uv run mypy .
    ```

3.  **Pre-Commit**: Ensure all checks pass before submitting changes.

## Roadmap

-   **Cycle 01**: Skeleton & Basic Loop (Completed)
-   **Cycle 02**: Pacemaker Trainer & Delta Learning
-   **Cycle 03**: Espresso Oracle & Self-Healing
-   **Cycle 04**: Adaptive Exploration
-   **Cycle 05**: Dynamics Engine & Uncertainty Watchdog
-   **Cycle 06**: Scale-up & Validation

## License

This project is licensed under the MIT License.
