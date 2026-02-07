# PYACEMAKER: Automated Active Learning for ML Potentials

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12+-blue)

**PYACEMAKER** is an autonomous "Self-Driving Laboratory" for creating state-of-the-art Machine Learning Interatomic Potentials (MLIP). By orchestrating Density Functional Theory (DFT) calculations, Active Learning (AL), and Molecular Dynamics (MD) validation into a single "Zero-Config" workflow, it allows researchers to generate robust potentials for complex alloys and interfaces.

## Key Features

*   **Zero-Config Workflow**: Define your elements and goals in a single YAML file.
*   **Modular Architecture**: Plug-and-play components for Oracle (DFT), Trainer (Pacemaker), and Dynamics (LAMMPS).
*   **Mock Mode**: Fully simulated active learning loop for rapid testing and development without heavy physics dependencies.
*   **Strict Validation**: Pydantic-based configuration and data models ensure robustness.
*   **Rich Logging**: Real-time console status updates and detailed file logging.

## Requirements

*   **Python**: 3.12 or higher
*   **Package Manager**: `uv` (recommended)

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-org/pyacemaker.git
    cd pyacemaker
    ```

2.  **Initialize Environment (using uv)**
    ```bash
    uv sync
    ```

## Usage

### 1. Create a Configuration File
Create a `config.yaml` file with your project settings. For a quick start using mocks:

```yaml
project_name: "my_first_project"
orchestrator:
  max_iterations: 5
oracle:
  type: "mock"
trainer:
  type: "mock"
dynamics:
  type: "mock"
structure_generator:
  type: "mock"
validator:
  type: "mock"
```

### 2. Run the Orchestrator
Execute the pipeline:

```bash
uv run python -m mlip_autopipec.main run config.yaml
```

The system will:
1.  Initialize components.
2.  Generate candidate structures.
3.  Run the Active Learning Loop (Label -> Train -> Validate -> Explore).
4.  Output results to the `active_learning/` directory.

## Architecture

```
src/mlip_autopipec/
├── config/             # Pydantic configuration models
├── domain_models/      # Data transfer objects (Structure, Potential)
├── infrastructure/     # Concrete implementations (Mock, etc.)
├── interfaces/         # Abstract Base Classes
├── orchestrator/       # Core logic and state machine
├── utils/              # Logging and helpers
└── main.py             # CLI entry point
```

## Roadmap

*   **Cycle 01 (Completed)**: Foundation, Orchestrator, Mocks, CLI.
*   **Cycle 02**: Oracle (Quantum Espresso) & Structure Generator.
*   **Cycle 03**: Trainer (Pacemaker) Integration.
*   **Cycle 04**: Dynamics Engine (LAMMPS) & Uncertainty.
*   **Cycle 05**: Scale-Up & Adaptive Policy.
*   **Cycle 06**: Production Readiness.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
