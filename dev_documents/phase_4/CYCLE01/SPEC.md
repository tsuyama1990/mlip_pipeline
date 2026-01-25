# Cycle 01 Specification: Core Framework & Configuration

## 1. Summary
This cycle establishes the foundational infrastructure of the **MLIP Auto PiPEC** system. It focuses on robust configuration management using Pydantic, centralized logging, and the persistence layer (DatabaseManager) that will serve as the backbone for the entire pipeline. The goal is to ensure that the system fails fast on invalid inputs and provides a thread-safe, consistent interface for data storage.

## 2. System Architecture

```ascii
mlip_autopipec/
├── __init__.py
├── app.py                      # CLI Entry Point (Stub)
├── config/
│   ├── __init__.py
│   ├── models.py               # **Aggregated Config Models**
│   └── schemas/
│       ├── __init__.py
│       ├── core.py             # **Core/Logging Config**
│       └── dft.py              # **DFT Config Schema**
├── data_models/
│   ├── __init__.py
│   └── atoms.py                # **ASE Atoms Pydantic Validator**
├── orchestration/
│   ├── __init__.py
│   └── database.py             # **DatabaseManager Class**
└── utils/
    ├── __init__.py
    └── logging.py              # **Logging Configuration**
```

**Key Modifications:**
- Create the package structure.
- Implement `mlip_autopipec/config/` for schema definitions.
- Implement `mlip_autopipec/orchestration/database.py` for ASE DB interactions.

## 3. Design Architecture

### 3.1. Configuration System (`config/`)
The system must be driven by a single YAML file, validated strictly before any heavy computation starts.
- **Domain Concepts**: `MLIPConfig` (Root), `DFTConfig`, `TrainingConfig`.
- **Validation**: Use Pydantic `v2`. Enforce `extra="forbid"` to catch typoed keys.
- **Type Safety**: All fields must have type hints. Units (eV, Angstrom) should be documented in docstrings.

### 3.2. Database Manager (`orchestration/database.py`)
A wrapper around `ase.db` (SQLite) to standardize how data is stored and retrieved.
- **Constraint**: Must handle concurrent write attempts (SQLite lock) gracefully using `tenacity` retries.
- **Invariants**: Every row must have a `status` (pending, running, completed, failed) and a `timestamp`.
- **Interface**:
  - `add_structure(atoms, metadata)`
  - `update_status(id, status)`
  - `get_pending_tasks()`

### 3.3. Atoms Validator (`data_models/atoms.py`)
A custom Pydantic type to validate ASE Atoms objects.
- **Validation**: Ensure input is an `ase.Atoms` instance. Check for `positions`, `numbers`, and `cell`.
- **Serialization**: Define how `Atoms` objects are serialized to/from Dict for Pydantic models (if needed) or rely on ASE's internal dict conversion.

## 4. Implementation Approach

1.  **Project Skeleton**: Run `uv init` or manual directory creation. Setup `pyproject.toml`.
2.  **Logging**: Create `utils/logging.py` using `rich` for pretty console output and file rotation.
3.  **Data Models**:
    - Implement `ASEAtoms` validator in `data_models/atoms.py`.
    - Create schemas in `config/schemas/*.py`.
    - Aggregate them in `config/models.py`.
4.  **Database**:
    - Implement `DatabaseManager` with context manager support (`with DatabaseManager(...) as db:`).
    - Add retry logic for `sqlite3.OperationalError`.
5.  **CLI Stub**: Create a basic `app.py` using `typer` that loads a config file and prints "Config Valid".

## 5. Test Strategy

### 5.1. Unit Testing
- **Config**: Test loading valid/invalid YAMLs. Assert `ValidationError` is raised for missing required fields.
- **Database**:
    - Test `add_structure` and retrieve it.
    - Test `update_status`.
    - **Concurrency Test**: Spawn threads trying to write to the same DB file to test locking resilience.
- **Atoms**: Pass valid/invalid objects to a model using `ASEAtoms`.

### 5.2. Integration Testing
- **Full Flow**:
    1.  Create a dummy `config.yaml`.
    2.  Run the CLI command to load it.
    3.  Initialize the database based on config paths.
    4.  Verify the DB file is created and has the correct schema/metadata.
