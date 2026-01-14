# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: The Foundation

This first cycle establishes the architectural backbone of the project. It includes:

- **Pydantic Schemas**: A robust, schema-driven framework for managing all system configurations.
- **DFT Factory Core**: The initial implementation of the `QEProcessRunner`, a component for interacting with the Quantum Espresso DFT engine.
- **Database Manager**: A data persistence layer for handling ASE-compatible databases with custom metadata.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```
