# Dependency Management

This project uses `uv` for fast and reliable dependency management.

## Managing Dependencies

### Adding a Dependency

To add a new package, use `uv add`:

```bash
uv add <package_name>
```

This updates `pyproject.toml` and `uv.lock`.

### Adding a Development Dependency

For tools like `pytest`, `ruff`, or `mypy` that are only needed for development:

```bash
uv add --dev <package_name>
```

### Syncing Environment

To ensure your virtual environment matches the lockfile:

```bash
uv sync
```

### Upgrading Dependencies

To upgrade packages:

```bash
uv lock --upgrade
uv sync
```

## Critical Dependencies

- **Core**:
    - `ase`: Atomic Simulation Environment (Structure manipulation).
    - `pydantic`: Data validation and settings management.
    - `numpy`: Numerical computing.
- **DFT**:
    - Quantum Espresso (`pw.x`) must be installed externally and accessible in `$PATH`.
- **ML**:
    - `mace-torch`: Foundation models.
    - `dscribe`: Descriptors (SOAP).
    - `scikit-learn`: Clustering and sampling.
- **Inference**:
    - `lammps`: MD engine (external binary required).
