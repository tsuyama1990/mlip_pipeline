
## Dependency Management

This project uses `uv` for dependency management.

### Installation

To install dependencies:
```bash
uv sync
```

### Adding Dependencies

To add a new dependency:
```bash
uv add <package_name>
```

### Updating Dependencies

To update dependencies:
```bash
uv lock --upgrade
```

### Running Tests

To run tests:
```bash
uv run pytest
```

### Dependency Categories (pyproject.toml)

The `pyproject.toml` file categorizes dependencies to clarify their purpose:

- **Core Data Models & Validation**: `pydantic`. Required for configuration and data schemas.
- **Scientific Computing**: `numpy`, `scipy`, `pandas`, `numba`. Required for numerical operations, spatial analysis (Voronoi), and performance.
- **Atomic Simulation Environment**: `ase`. The core library for atomic structures and file I/O.
- **Machine Learning / Surrogate**: `mace-torch`, `dscribe`, `torch`. Required for the Surrogate Module (Cycle 03) to perform pre-screening and featurization.
- **CLI & Output**: `typer`, `rich`, `pyyaml`. For the command-line interface and configuration parsing.
- **Distributed Computing**: `dask`, `distributed`, `tenacity`. For managing the workflow execution and retries.
- **Visualization**: `plotly`, `matplotlib`. For dashboarding and analysis.
