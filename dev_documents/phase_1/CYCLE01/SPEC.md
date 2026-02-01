# Cycle 01: Foundation & Core Models

## 1. Summary

Cycle 01 is the "Big Bang" of the PyAceMaker project. Before any physics can be simulated or any machine learning models trained, we must establish a rigid, type-safe foundation. In many scientific software projects, data structures are often ad-hoc dictionaries or loose arrays, leading to "spaghetti code" where it is unclear what keys exist or what shape an array should have. This project rejects that approach in favour of a **Schema-First Design**.

The primary objective of this cycle is to define the **Domain Models** that will serve as the ubiquitous language of the system. We will utilise `pydantic` to enforce strict validation rules. For example, a `Structure` object must not just hold atomic positions; it must verify that the number of positions matches the number of chemical symbols, and that the cell is a 3x3 matrix. If these invariants are violated, the system should crash immediately and informatively, rather than propagating corrupt data downstream.

In addition to data models, we will implement the infrastructure for **Configuration Management** and **Logging**. A complex automated pipeline needs to be configurable without code changes. We will implement a YAML loader that maps directly to our Pydantic `Config` schema, ensuring that the user's `config.yaml` is validated at startup. Similarly, we will set up a structured logging system using the `rich` library, providing both human-readable console output and machine-parsable file logs, which are crucial for debugging long-running autonomous jobs.

By the end of this cycle, we will not have a running physical simulation, but we will have a compiled, type-checked codebase that can load a configuration, validate atomic structures, and log its own existence. This is the bedrock upon which the Oracle, Trainer, and Explorer will be built.

## 2. System Architecture

The architecture for this cycle focuses on the `domain_models` and `infrastructure` layers. The Orchestrator and Physics modules will be added in subsequent cycles.

### File Structure
The files to be created or modified in this cycle are highlighted in **bold**.

```ascii
mlip_autopipec/
├── pyproject.toml
├── src/
│   └── mlip_autopipec/
│       ├── **__init__.py**
│       ├── **app.py**                  # CLI Entry point (Skeleton)
│       ├── domain_models/              # Pydantic Schemas
│       │   ├── **__init__.py**
│       │   ├── **structure.py**        # Atom & Structure definitions
│       │   └── **config.py**           # Global Configuration Schema
│       └── infrastructure/
│           ├── **__init__.py**
│           ├── **logging.py**          # Rich logging setup
│           └── **io.py**               # YAML/JSON/Pickle handlers
└── tests/
    ├── **conftest.py**
    └── domain_models/
        └── **test_structure.py**       # Tests for Structure invariants
```

### Component Interaction
1.  **User** provides `config.yaml`.
2.  **`app.py`** calls `infrastructure.io.load_config`.
3.  **`infrastructure.io`** reads YAML and passes dict to **`domain_models.config.Config`**.
4.  **`Config`** validates types (e.g., ensuring `cutoff` is positive).
5.  **`infrastructure.logging`** is initialised based on config verbosity.

## 3. Design Architecture

This section details the Pydantic schemas that constitute the "DNA" of the system.

### 3.1. Structure Domain Model (`domain_models/structure.py`)
The `Structure` class is the most fundamental object, representing a collection of atoms.

-   **Class `Structure` (Pydantic Model)**:
    -   **Fields**:
        -   `symbols`: `List[str]` (e.g., `["Ti", "O"]`).
        -   `positions`: `NDArray[Shape["*, 3"], Float]` (Numpy array of coordinates).
        -   `cell`: `NDArray[Shape["3, 3"], Float]` (Lattice vectors).
        -   `pbc`: `Tuple[bool, bool, bool]` (Periodic boundary conditions).
        -   `properties`: `Dict[str, Any]` (Flexible storage for energy, forces).
    -   **Validators**:
        -   `check_consistency`: Ensures `len(symbols) == len(positions)`.
        -   `check_cell_shape`: Ensures `cell` is exactly 3x3.
    -   **Methods**:
        -   `from_ase(atoms: ase.Atoms) -> Structure`: Factory method.
        -   `to_ase() -> ase.Atoms`: Conversion method.
        -   `get_chemical_formula() -> str`.

### 3.2. Configuration Domain Model (`domain_models/config.py`)
The configuration hierarchy controls the behaviour of all subsystems.

-   **Class `Config`**:
    -   **Fields**:
        -   `project_name`: `str`.
        -   `logging`: `LoggingConfig`.
        -   `orchestrator`: `OrchestratorConfig`.
        -   `potential`: `PotentialConfig`.

-   **Class `LoggingConfig`**:
    -   `level`: `Literal["DEBUG", "INFO", "WARNING", "ERROR"]`.
    -   `file_path`: `Path`.

-   **Class `PotentialConfig`**:
    -   `elements`: `List[str]`.
    -   `cutoff`: `float` (Must be > 0).
    -   `seed`: `int`.

### 3.3. Infrastructure (`infrastructure/logging.py`, `io.py`)
-   **Logging**:
    -   Uses `rich.console.Console` for pretty printing.
    -   Uses standard `logging.FileHandler` for persistent logs.
    -   Function `setup_logging(config: LoggingConfig)` configures the root logger.
-   **IO**:
    -   `load_yaml(path: Path) -> Dict`: Safe YAML loading.
    -   `dump_yaml(data: Dict, path: Path)`: YAML dumping.

## 4. Implementation Approach

This cycle will be implemented in a strict sequence to ensure dependencies are met.

### Step 1: Project Skeleton
-   Create the directory structure.
-   Create empty `__init__.py` files to make packages importable.

### Step 2: Infrastructure - IO & Logging
-   Implement `src/mlip_autopipec/infrastructure/logging.py`.
    -   Define the `RichHandler` and `FileHandler`.
-   Implement `src/mlip_autopipec/infrastructure/io.py`.
    -   Use `pyyaml` (safe load).
    -   Add error handling for `FileNotFoundError`.

### Step 3: Domain Models - Config
-   Implement `src/mlip_autopipec/domain_models/config.py`.
    -   Define nested models first (`LoggingConfig`, `PotentialConfig`).
    -   Assemble them into the main `Config` model.
    -   Add `Config.from_yaml(path)` factory method using `infrastructure.io`.

### Step 4: Domain Models - Structure
-   Implement `src/mlip_autopipec/domain_models/structure.py`.
    -   This requires `numpy` and `pydantic`.
    -   Implement the `arbitrary_types_allowed=True` config in Pydantic to support Numpy arrays.
    -   Implement the validators.

### Step 5: Application Entry Point
-   Implement `src/mlip_autopipec/app.py`.
    -   Use `typer` to define a CLI command `init` (creates a sample config) and `check` (loads and validates config).

## 5. Test Strategy

### 5.1. Unit Testing
We will use `pytest` to rigorously test the constraints.

-   **Test `Structure` Validation**:
    -   Create a test that tries to initialise a `Structure` with 2 symbols but 3 positions. Assert that `ValidationError` is raised.
    -   Create a test with a 2x2 cell matrix. Assert `ValidationError`.
    -   Test `from_ase` and `to_ase` round-trip conversion to ensure data integrity.

-   **Test `Config` Loading**:
    -   Create a temporary `config.yaml` with valid data. Assert `Config.from_yaml` returns a valid object.
    -   Create a `config.yaml` with a negative cutoff. Assert `ValidationError`.
    -   Create a `config.yaml` with missing required fields. Assert `ValidationError`.

### 5.2. Integration Testing
-   **CLI Test**:
    -   Invoke the `app.py` CLI using `typer.testing.CliRunner`.
    -   Run `mlip-auto init` and check if `config.yaml` is created.
    -   Run `mlip-auto check` and verify the exit code is 0.

### 5.3. Pre-commit
-   Run `mypy` on the new files. Since we use `pydantic`, type checking should be strict.
-   Run `ruff` to ensure formatting.
