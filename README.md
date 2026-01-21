# MLIP-AutoPipe

MLIP-AutoPipe is a project designed to provide a "Zero-Human" protocol for the autonomous generation of Machine Learning Interatomic Potentials (MLIPs). This system automates the entire MLIP lifecycle, from initial data generation to active learning and large-scale production simulations.

## System Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed diagram and module description.

## Dependencies

The project requires Python >= 3.11. Key dependencies include:
- `ase`
- `numpy`
- `pydantic`
- `dask`
- `matplotlib`
- `mace-torch`
- `dscribe`

See `pyproject.toml` for the full list and version constraints.

## Dependency Management

We use `uv` for dependency management.
- **Strict Versions**: All dependencies in `pyproject.toml` must have version constraints (e.g., `>=3.11`, `==1.0.0`).
- **Locking**: `uv.lock` is the source of truth for reproducible installs.
- **Updates**: Run `uv sync` to update the environment. Major dependency upgrades should be tested in a separate branch.
- **Conflict Resolution**: If conflicts arise (e.g. `icet`), downgrade Python version constraints or pin specific package versions, but prefer `uv` resolution.

## Cycles

### Cycle 01: Core Framework & User Interface
- **Strict Schema**: `MinimalConfig` & `SystemConfig`.
- **Data Persistence**: `DatabaseManager` (ASE-db).
- **CLI Initialization**.

### Cycle 02: Automated DFT Factory
- **Autonomous Execution**: `QERunner`.
- **Auto-Recovery**: `RecoveryHandler`.

### Cycle 03: Structure Generator
- **Generators**: SQS, NMS, Defects.

### Cycle 04: Surrogate Explorer
- **Pre-screening**: MACE foundation model.
- **Selection**: FPS on SOAP descriptors.

### Cycle 05: Active Learning & Training
- **Training**: Pacemaker integration.
- **Physics**: ZBL baseline subtraction.

### Cycle 06: Scalable Inference Engine (Part 1)
- **MD**: LAMMPS integration.
- **UQ**: Uncertainty monitoring.

### Cycle 07: Scalable Inference Engine (Part 2)
- **Extraction**: Periodic embedding of local clusters.
- **Masking**: Force masking for training.

### Cycle 08: Orchestration
- **WorkflowManager**: State machine for autonomous operation.
- **TaskQueue**: Distributed execution via Dask.
- **Dashboard**: HTML monitoring interface.

## Getting Started

Create environment:
```bash
uv sync --extra dev
```

Run:
```bash
# Initialize a new project
mlip-auto init

# Validate your configuration
mlip-auto check-config input.yaml

# Initialize the database
mlip-auto db init

# Run the pipeline (if implemented)
mlip-auto run input.yaml
```
