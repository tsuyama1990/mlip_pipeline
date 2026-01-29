# PYACEMAKER: Automated MLIP Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**PYACEMAKER** is a fully automated, "Zero-Config" system for constructing and operating State-of-the-Art Machine Learning Interatomic Potentials (MLIP) using the Pacemaker (ACE) engine. It democratizes computational materials science by enabling researchers to generate robust, physics-informed potentials without needing deep expertise in machine learning or DFT.

## Features

*   **Structure Generation**: Automated generation of initial atomic structures ("Cold Start") and random perturbations for candidate exploration.
*   **Project Initialization**: Quickly set up a new active learning project with a template configuration using `mlip-auto init`.
*   **Robust Configuration**: Strict schema validation ensures your settings are correct before any expensive calculations start.
*   **Workflow Orchestration**: Automated state management tracks the progress of the active learning loop, handling persistence and recovery.
*   **Modular Architecture**: Designed for extensibility, separating domain models, infrastructure, and orchestration logic.

## Requirements

*   **Python**: 3.11 or higher
*   **Package Manager**: `uv` (recommended) or `pip`
*   **Dependencies**: `pydantic`, `typer`, `rich`, `ase`, `pyyaml`, `numpy`

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/mlip-autopipec.git
    cd mlip-autopipec
    ```

2.  **Install dependencies**:
    Using `uv`:
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -e .[dev]
    ```

## Usage

### Quick Start

1.  **Initialize a new project**:
    ```bash
    mlip-auto init
    ```
    This creates a `config.yaml` in your current directory.

2.  **Validate Configuration**:
    ```bash
    mlip-auto check --config config.yaml
    ```

3.  **Run the Active Learning Loop**:
    ```bash
    mlip-auto run-loop
    ```
    The system will initialize the workflow and track the state in `workflow_state.json`.

## Architecture Structure

```ascii
src/mlip_autopipec/
├── app.py                      # CLI Entry Point
├── constants.py                # Global Constants
├── cli/                        # CLI Command Implementations
├── domain_models/              # Pydantic Schemas (Config, Structure, Workflow)
├── infrastructure/             # Logging and I/O
└── orchestration/              # Workflow Management
```

## Roadmap

*   **Structure Generation**: (Completed) Algorithms for generating initial and candidate structures.
*   **DFT Oracle**: Automated Quantum Espresso calculations.
*   **Training Loop**: Integration with Pacemaker for potential fitting.
*   **Active Learning**: MD-based sampling and uncertainty quantification.
*   **Validation**: Physical validation tests (Phonons, Elasticity).

## License

This project is licensed under the MIT License.
