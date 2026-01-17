# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle lays the bedrock for the entire MLIP-AutoPipe ecosystem. Before any physics can be simulated or any potentials trained, we must establish a robust, type-safe, and verifiable infrastructure. The primary objective of this cycle is to implement the **Configuration Management System** and the **Core Utilities** (Database and Logging) that will serve as the nervous system of the application.

### Key Features:
- **Strict Schema**: A rigorous Pydantic V2 schema defines the `MinimalConfig` (user input) and `SystemConfig` (internal state), ensuring type safety and validity before any computation begins.
- **Data Persistence**: A `DatabaseManager` wrapper around `ase.db` ensures that no data is saved without its provenance metadata, enforcing traceability.
- **Centralized Logging**: A unified logging system provides structured, machine-parsable logs, essential for debugging autonomous workflows.
- **CLI Initialization**: The `mlip-auto run input.yaml` command initializes a project workspace, validates the configuration, and establishes the database connection.

## Cycle 02: Automated DFT Factory

This cycle implements the **Automated DFT Factory**, the "engine room" of the pipeline. It autonomously manages Quantum Espresso calculations, ensuring robust data generation for MLIP training.

### Key Features:
- **Autonomous Execution**: The `QERunner` manages the entire lifecycle of a DFT calculation, from input generation to output parsing.
- **Auto-Recovery**: A state-machine-based `RecoveryHandler` automatically detects and fixes common DFT errors (convergence failure, diagonalization errors) by adjusting parameters like mixing beta and temperature.
- **Physics-Informed Input Generation**: The `InputGenerator` automatically selects appropriate pseudopotentials (SSSP) and K-point grids based on cell density and material composition (e.g., handling magnetism for Fe, Co, Ni).
- **Data Standardization**: All results are encapsulated in a strictly typed `DFTResult` object, ensuring consistency across the pipeline.

## Development Status

Cycle 01 and Cycle 02 features are implemented.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv sync --extra dev
```

Run the initialization:

```bash
mlip-auto run input.yaml
```
