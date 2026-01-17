# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle lays the bedrock for the entire MLIP-AutoPipe ecosystem. Before any physics can be simulated or any potentials trained, we must establish a robust, type-safe, and verifiable infrastructure. The primary objective of this cycle is to implement the **Configuration Management System** and the **Core Utilities** (Database and Logging) that will serve as the nervous system of the application.

### Key Features:
- **Strict Schema**: A rigorous Pydantic V2 schema defines the `MinimalConfig` (user input) and `SystemConfig` (internal state), ensuring type safety and validity before any computation begins.
- **Data Persistence**: A `DatabaseManager` wrapper around `ase.db` ensures that no data is saved without its provenance metadata, enforcing traceability.
- **Centralized Logging**: A unified logging system provides structured, machine-parsable logs, essential for debugging autonomous workflows.
- **CLI Initialization**: The `mlip-auto run input.yaml` command initializes a project workspace, validates the configuration, and establishes the database connection.

## Development Status

The project is currently implementing Cycle 01 features.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv sync --extra dev
```

Run the initialization:

```bash
mlip-auto run input.yaml
```
