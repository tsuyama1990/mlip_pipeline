# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## Cycle 01: Core Framework & User Interface

This cycle establishes the bedrock for the MLIP-AutoPipe ecosystem. It implements the Configuration Management System, Centralized Logging, and Database Interfaces that serve as the nervous system of the application.

### Key Features:
- **Strict Schema Configuration**: The system uses rigorous Pydantic V2 models (`MinimalConfig`, `SystemConfig`) to validate user inputs (`input.yaml`) immediately, ensuring type safety and preventing silent failures.
- **Config Factory**: A robust factory pattern that expands a minimal user configuration into a fully resolved system state, handling directory creation and path resolution automatically.
- **Database Interface**: A `DatabaseManager` wrapping `ase.db` enforces schema compliance and metadata provenance, ensuring every calculation is traceable to its source configuration.
- **Centralized Logging**: A structured logging system using `rich` provides human-readable console output and machine-parsable file logs.
- **CLI Entry Point**: The `mlip-auto` command-line tool (built with `Typer`) allows users to initialize and run projects with a simple command.

## Cycle 02: Automated DFT Factory (Planned)

This cycle will implement the cornerstone of the physics engine: a robust and autonomous DFT calculation factory using Quantum Espresso.

## Cycle 03: Physics-Informed Generator (Planned)

This cycle will implement the structure generation module (SQS, NMS, Defects).

## Cycle 04: Surrogate Explorer (Planned)

This cycle will introduce the MACE surrogate model and Farthest Point Sampling (FPS) for efficient structure selection.

## Cycle 05: Active Learning & Training (Planned)

This cycle will automate the training of ACE potentials using Pacemaker.

## Cycle 06: Scalable Inference Engine (Planned)

This cycle will implement the LAMMPS runner and extrapolation grade monitoring.

## Cycle 07: Scalable Inference Part 2 (Planned)

This cycle will handle periodic embedding and force masking for extracting training data from MD.

## Cycle 08: Orchestration (Planned)

This cycle will integrate all modules into a cohesive, scalable system using Dask.

## Getting Started

To get started with the project, create a virtual environment and install the dependencies:

```bash
uv venv
uv pip install -e ".[dev]"
```

Run the workflow:

```bash
mlip-auto run input.yaml
```
