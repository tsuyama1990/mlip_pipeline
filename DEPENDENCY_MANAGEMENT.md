
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
