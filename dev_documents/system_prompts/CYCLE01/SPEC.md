# Cycle 01 Specification: Core Infrastructure

## 1. Summary
Cycle 01 focuses on establishing the foundational infrastructure of the PYACEMAKER system. This cycle is critical as it sets the standards for code quality, configuration management, logging, and modular extensibility that will be used throughout the project.

The primary deliverables are:
1.  **Project Skeleton**: A robust Python package structure with strict typing and linter configurations.
2.  **Configuration Management**: A Pydantic-based configuration loader that parses `config.yaml`, validates data types, and provides sensible defaults. This ensures that the "Zero-Config" goal is built on a solid schema.
3.  **Centralized Logging**: A uniform logging system that captures standard output, errors, and debug information, capable of writing to both console and file.
4.  **CLI Entry Point**: A command-line interface (CLI) using `typer` to launch the application.
5.  **Abstract Base Classes (ABCs)**: The `BaseModule` interface that all future modules (Oracle, Trainer, etc.) will implement, ensuring a consistent API for the Orchestrator.

## 2. System Architecture

The following file structure must be implemented. **Bold files** are the ones to be created or significantly modified in this cycle.

```text
src/
└── **pyacemaker/**
    ├── **__init__.py**
    ├── **main.py**             # CLI Entry Point
    └── **core/**
        ├── **__init__.py**
        ├── **base.py**         # Abstract Base Classes (BaseModule)
        ├── **config.py**       # Pydantic Configuration Models
        ├── **logging.py**      # Centralized Logging Setup
        └── **exceptions.py**   # Custom Exception Classes
```

### File Details
-   `src/pyacemaker/main.py`: The entry point script. It uses `typer` to define commands like `run`, `validate-config`.
-   `src/pyacemaker/core/config.py`: Defines the `PYACEMAKERConfig` class using Pydantic. It handles YAML loading and validation.
-   `src/pyacemaker/core/logging.py`: Configures the Python `logging` library. It should support different log levels (INFO, DEBUG) based on the CLI verbosity flag.
-   `src/pyacemaker/core/base.py`: Defines `BaseModule` using `abc.ABC`. It mandates an `execute()` method for all subsystems.
-   `src/pyacemaker/core/exceptions.py`: Defines `PYACEMAKERError`, `ConfigurationError`, and other base exceptions.

## 3. Design Architecture

### 3.1. Configuration Schema (Pydantic)
The system's behavior is driven by a single `config.yaml`. We model this using Pydantic `BaseModel`.

```python
from pydantic import BaseModel, Field
from pathlib import Path

class ProjectConfig(BaseModel):
    name: str
    root_dir: Path

class DFTConfig(BaseModel):
    code: str = "quantum_espresso"
    # ... other DFT settings (placeholders for now)

class PYACEMAKERConfig(BaseModel):
    project: ProjectConfig
    dft: DFTConfig
    # ... placeholders for other sections
```
**Constraints & Invariants:**
-   `root_dir` must be converted to an absolute `Path` object upon loading.
-   Missing required fields must raise a clear `ValidationError`.

### 3.2. Module Interface (Abstract Base Class)
To ensure the Orchestrator can treat all components uniformly, we define a strict interface.

```python
from abc import ABC, abstractmethod

class BaseModule(ABC):
    def __init__(self, config: PYACEMAKERConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> dict:
        """Executes the module's main logic."""
        pass
```

## 4. Implementation Approach

### Step 1: Package Structure & Linter Setup
-   Create the directory structure `src/pyacemaker/core`.
-   Verify that `pyproject.toml` (created in the root) is respected by `ruff` and `mypy`.

### Step 2: Custom Exceptions
-   Implement `src/pyacemaker/core/exceptions.py`.
-   Define `PYACEMAKERError` inheriting from `Exception`.
-   Define `ConfigError` inheriting from `PYACEMAKERError`.

### Step 3: Logging System
-   Implement `src/pyacemaker/core/logging.py`.
-   Create a function `setup_logging(level: str)` that configures the root logger with a standardized formatter (Timestamp - Name - Level - Message).

### Step 4: Configuration Manager
-   Implement `src/pyacemaker/core/config.py`.
-   Define the Pydantic models.
-   Add a `load_config(path: str) -> PYACEMAKERConfig` function that reads YAML and returns the validated model.

### Step 5: Abstract Base Classes
-   Implement `src/pyacemaker/core/base.py`.
-   Define `BaseModule` as described in Design Architecture.

### Step 6: CLI Entry Point
-   Implement `src/pyacemaker/main.py` using `typer`.
-   Create a command `run(config_path: str)` that:
    1.  Sets up logging.
    2.  Loads the config.
    3.  Prints "Configuration loaded successfully" (as a placeholder for actual execution).

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Testing (`tests/core/test_config.py`)**:
    -   Create a minimal valid `config.yaml`. Assert it loads correctly.
    -   Create a malformed `config.yaml` (missing fields). Assert `ValidationError` or `ConfigError` is raised.
-   **Logging Testing (`tests/core/test_logging.py`)**:
    -   Verify that `get_logger` returns a logger instance.
    -   (Optional) Capture logs (using `pytest`'s `caplog`) to verify format.

### 5.2. Integration Testing
-   **CLI Smoke Test (`tests/test_main.py`)**:
    -   Use `typer.testing.CliRunner` to invoke the `run` command.
    -   Verify exit code 0 on valid input.
    -   Verify exit code 1 on missing file.
