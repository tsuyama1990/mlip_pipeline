
## Dependencies

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

### Running Tests

To run the test suite:

```bash
uv run pytest
```

### Dependency Groups

The `pyproject.toml` file organizes dependencies into logical groups:
- **Core**: Pydantic, NumPy, SciPy, Pandas
- **ASE**: Atomic Simulation Environment
- **Machine Learning**: PyTorch, MACE, Dscribe
- **Distributed**: Dask
- **CLI**: Typer, Rich
- **Dev**: Pytest, Ruff, Mypy
