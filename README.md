# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle establishes the structural foundation of the MLIP-AutoPipe project. It implements the rigorous configuration management system, the centralized logging infrastructure, and the database schema for provenance tracking.

### Key Features:
- **Schema-First Configuration**: A strictly typed configuration system using `Pydantic`, ensuring that all user inputs (from `input.yaml`) are validated before execution.
- **Project Initialization**: A new `mlip-auto init` command that bootstraps the project directory, initializes the SQLite database with metadata, and sets up logging.
- **Database Wrapper**: A robust `DatabaseManager` that wraps `ase.db`, enforcing the storage of system configuration and calculation provenance (metadata) for every entry.
- **CLI Foundation**: The entry point for the application using `Typer`, providing a user-friendly command-line interface.

## Future Cycles (Planned)

- **Cycle 02: Automated DFT Factory**: Autonomous DFT calculations with error recovery.
- **Cycle 03: Physics-Informed Generator**: Structure generation (SQS, NMS).
- **Cycle 04: Surrogate Explorer**: Pre-screening with MACE and FPS.
- **Cycle 05: Active Learning & Training**: Automated potential training.
- **Cycle 06: Scalable Inference Engine - Part 1**: MD simulations and uncertainty quantification.
- **Cycle 07: Scalable Inference Engine - Part 2**: Periodic embedding and force masking.
- **Cycle 08: Orchestration & Production Readiness**: Full workflow orchestration.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

Initialize a new project:

```bash
mlip-auto init input.yaml
```

Run the workflow (Configured project):

```bash
mlip-auto run input.yaml
```

Check status:

```bash
mlip-auto status .
```
