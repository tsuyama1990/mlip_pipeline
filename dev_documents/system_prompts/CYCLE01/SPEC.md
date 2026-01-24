# Cycle 01 Specification: Core Framework & Data Models

## 1. Summary

Cycle 01 marks the foundational phase of the **MLIP-AutoPipe** project. The primary objective of this cycle is to establish the robust skeletal infrastructure required to support the complex "Zero-Human" active learning pipeline. Unlike traditional scripts that often suffer from "spaghetti code" and loose typing, MLIP-AutoPipe prioritises rigid structural integrity, type safety, and clear separation of concerns from Day One.

In this cycle, we focus on three critical pillars: **Configuration Management**, **Data Modeling**, and **Persistence**.

Firstly, we will implement a strictly typed configuration system using Pydantic. The `input.yaml` file provided by the user will not be parsed as a raw dictionary but will be validated against a rigorous schema (`MLIPConfig`). This ensures that errors such as missing keys, invalid types (e.g., string instead of float), or physically nonsensical values (e.g., negative temperature) are caught immediately at startup, rather than causing obscure crashes deep in a 10-hour simulation. This configuration system will be hierarchical, separating `DFTConfig`, `TrainingConfig`, and `WorkflowConfig`.

Secondly, we will define the core domain models. The central data object in materials science is the atomic structure. We will implement a custom Pydantic type, `ASEAtoms`, which wraps the standard `ase.Atoms` object. This wrapper will enforce validation rules—ensuring that every structure has valid positions, atomic numbers, and cell dimensions—and facilitate seamless serialisation/deserialisation. This solves a common pain point where "bad structures" (e.g., overlapping atoms, zero-volume cells) silently propagate through pipelines.

Thirdly, we will build the `DatabaseManager`. This module serves as the persistence layer and the communication hub between future asynchronous workers. We will wrap the standard `ase.db` (SQLite) with a context manager that enforces schema consistency. Unlike a raw ASE database, our manager will handle metadata standardisation, ensuring that every entry has a unique ID, a status flag (pending, running, completed), and provenance info. This transforms the database from a simple storage file into a robust priority queue that will drive the active learning loop.

By the end of Cycle 01, we will have a compiled, installable package where a user can initialise the database and validate their input configuration, laying the groundwork for the heavy computational engines (DFT, MD) to be added in subsequent cycles.

## 2. System Architecture

This section details the file structure and code blueprints for Cycle 01. Files marked in **bold** are to be created or modified in this cycle.

### 2.1. File Structure

```ascii
mlip_autopipec/
├── **pyproject.toml**              # Project dependencies and tool config
├── **README.md**                   # Project documentation
├── **src/**
│   └── **mlip_autopipec/**
│       ├── **__init__.py**         # Top-level package export
│       ├── **app.py**              # CLI Entrypoint (Skeleton)
│       ├── **config/**
│       │   ├── **__init__.py**
│       │   ├── **models.py**       # Aggregated Configuration Models
│       │   └── **schemas/**
│       │       ├── **__init__.py**
│       │       ├── **dft.py**      # DFT Configuration Schema
│       │       ├── **training.py** # Training Configuration Schema
│       │       └── **workflow.py** # Workflow/Orchestration Schema
│       ├── **data_models/**
│       │   ├── **__init__.py**
│       │   ├── **types.py**        # Custom Pydantic Types (ASEAtoms)
│       │   ├── **status.py**       # JobStatus Enums
│       │   └── **manager.py**      # DatabaseManager Implementation
│       └── **utils/**
│           ├── **__init__.py**
│           └── **logging.py**      # Centralised Logging Setup
└── **tests/**
    ├── **__init__.py**
    ├── **conftest.py**             # Pytest Fixtures (Mock DB, Config)
    ├── **test_config.py**          # Configuration Tests
    └── **test_db_manager.py**      # Database Manager Tests
```

### 2.2. Code Blueprints

#### `src/mlip_autopipec/config/schemas/dft.py`
This file defines the parameters for Quantum Espresso.

```python
from pydantic import BaseModel, Field
from typing import Literal

class DFTConfig(BaseModel):
    command: str = Field(..., description="MPI command to run QE, e.g., 'mpirun -np 32 pw.x'")
    pseudopotential_dir: str = Field(..., description="Path to SSSP directory")
    ecutwfc: float = Field(default=60.0, ge=20.0, description="Wavefunction cutoff (Ry)")
    kspacing: float = Field(default=0.15, description="K-point grid density (1/A)")
    smearing: Literal['mv', 'gauss'] = 'mv'
    degauss: float = 0.02
```

#### `src/mlip_autopipec/data_models/types.py`
This file implements the `ASEAtoms` type validator.

```python
from typing import Any, Annotated
from ase import Atoms
from pydantic import BeforeValidator

def validate_atoms(v: Any) -> Atoms:
    if isinstance(v, Atoms):
        return v
    # Logic to convert dict or json back to Atoms if needed
    raise ValueError("Input must be an ASE Atoms object")

ASEAtoms = Annotated[Atoms, BeforeValidator(validate_atoms)]
```

#### `src/mlip_autopipec/data_models/manager.py`
The Database Manager handles `ase.db` interactions safely.

```python
from ase.db import connect
from pathlib import Path
from contextlib import contextmanager
from .status import JobStatus

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def connection(self):
        # Handles opening and closing connection
        con = connect(self.db_path)
        try:
            yield con
        finally:
            pass # ase.db handles closure usually

    def initialize(self):
        # Set up metadata tables if needed
        pass

    def add_structure(self, atoms, status: JobStatus = JobStatus.PENDING, **kwargs):
        with self.connection() as db:
            db.write(atoms, status=status.value, **kwargs)
```

#### `src/mlip_autopipec/config/models.py`
The master configuration model that aggregates everything.

```python
from pydantic import BaseModel
from .schemas.dft import DFTConfig
from .schemas.training import TrainingConfig

class MLIPConfig(BaseModel):
    project_name: str
    dft: DFTConfig
    training: TrainingConfig
    # ... other fields
```

## 3. Design Architecture

The design architecture of Cycle 01 is centred around the **Single Source of Truth** principle.

### 3.1. Domain Concepts

1.  **Configuration as Code**: The `MLIPConfig` object is the runtime representation of the user's intent. By defining it with Pydantic, we treat configuration with the same rigour as code. It acts as an immutable reference passed down to the Generator, Runner, and Trainer modules.
    -   *Constraint*: Extra fields in YAML are forbidden (`extra="forbid"`) to prevent users from thinking they are setting a parameter that is actually ignored.
    -   *Validation*: Physical constraints (e.g., Cutoff > 0) are enforced at the schema level, not in business logic.

2.  **The "Enriched" Atom**: A standard `ase.Atoms` object is insufficient for our pipeline. We need to attach metadata: `status`, `uncertainty_grade`, `provenance` (e.g., "generated by SQS" or "extracted from MD").
    -   *Design Choice*: We do not subclass `ase.Atoms` because it complicates serialisation. Instead, we use `ase.db`'s key-value pairs to store this metadata alongside the structure. The `DatabaseManager` abstracts this, providing methods like `get_pending_tasks()` which internally queries `status="pending"`.

3.  **Job Status Lifecycle**:
    -   `PENDING`: Structure generated, waiting for DFT.
    -   `RUNNING`: Picked up by a worker.
    -   `COMPLETED`: DFT finished, forces parsed.
    -   `FAILED`: DFT crashed (potentially recoverable).
    -   `ARCHIVED`: Structure invalid or duplicate.
    -   This lifecycle is defined in `status.py` as a Python Enum, ensuring type safety when querying the DB.

### 3.2. Consumers and Producers

-   **Producers**: In Cycle 02, the Generator will produce `PENDING` atoms. In Cycle 07, the Inference Engine will produce `PENDING` candidates.
-   **Consumers**: The `DatabaseManager` is the primary consumer of raw atoms and the producer of "Managed Atoms" (atoms + ID + status). The CLI (`app.py`) consumes `input.yaml` and produces the `MLIPConfig` object.

## 4. Implementation Approach

The implementation will proceed in a bottom-up fashion.

### Step 1: Environment & Project Skeleton
We will start by initialising the project structure. This involves creating the directories defined in the System Architecture. We will also configure `pyproject.toml` with the necessary dependencies: `ase`, `pydantic`, `typer`, `numpy`.

### Step 2: Configuration Schemas
We will implement the Pydantic models. We start with the leaf nodes (`dft.py`, `training.py`) and then implement the root `models.py`.
-   **Task**: Implement `DFTConfig` with validators for `ecutwfc` (must be > 0) and `pseudopotential_dir` (must exist).
-   **Task**: Implement `TrainingConfig` (placeholder for now, but strict types).
-   **Task**: Create a sample `input.yaml` and write a test script to parse it.

### Step 3: Core Data Models
We will implement `src/mlip_autopipec/data_models`.
-   **Task**: Define `JobStatus` Enum.
-   **Task**: Implement `ASEAtoms` type. This requires understanding Pydantic v2's `Annotated` validators. We need to ensure that when Pydantic sees an `Atoms` object, it validates it, and when it sees a dict (from JSON), it attempts to convert it to `Atoms`.

### Step 4: Database Manager
This is the most complex part of Cycle 01.
-   **Task**: Implement `DatabaseManager` class.
-   **Task**: Implement `initialize()` to create the DB file.
-   **Task**: Implement `count()`, `write()`, `update_status()`.
-   **Task**: Ensure thread-safety considerations (SQLite has limitations, but ASE handles file locking reasonably well for moderate concurrency).

### Step 5: CLI Entrypoint
-   **Task**: Implement a basic `app.py` using `typer`.
-   **Task**: Add a command `init` that generates a default `input.yaml`.
-   **Task**: Add a command `validate` that reads `input.yaml` and prints "Configuration Valid" or the error trace.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)
Unit tests in Cycle 01 will focus on **Schema Validation** and **Data Integrity**. Since we are building the foundation, these tests must be exhaustive.

-   **Config Tests**: We will create a suite of tests for `MLIPConfig`.
    -   *Valid Case*: Load a perfect YAML file. Assert all fields match.
    -   *Missing Field*: Remove `dft.command`. Assert `ValidationError` is raised.
    -   *Type Mismatch*: Set `dft.ecutwfc` to "sixty". Assert `ValidationError`.
    -   *Logic Error*: Set `dft.ecutwfc` to -10. Assert `ValidationError` (via Pydantic `gt=0` constraint).
-   **Type Tests**: We will test `ASEAtoms`.
    -   Pass a valid `ase.Atoms` object -> Success.
    -   Pass a generic object -> Failure.
    -   Pass a dict representation -> Verify it converts back to Atoms (if implemented) or fails gracefully.
-   **DB Manager Tests**: We will mock `ase.db.connect` or use a temporary file.
    -   Test `add_structure`: Verify the structure is in the DB and has `status="pending"`.
    -   Test `update_status`: Change status to `COMPLETED` and verify the change persists.

### 5.2. Integration Testing Approach (Min 300 words)
Integration tests will verify the interaction between the Configuration loader and the Database Manager.

-   **Workflow Init Test**:
    1.  Create a temporary directory.
    2.  Write a valid `input.yaml`.
    3.  Run the `app.py validate` command (programmatically invoking the Typer app).
    4.  Assert exit code is 0.
-   **Persistence Round-Trip**:
    1.  Initialize `DatabaseManager` with a temp SQLite file.
    2.  Create an `Atoms` object (e.g., a simple H2 molecule).
    3.  Save it via the manager.
    4.  Close the manager.
    5.  Re-open the manager.
    6.  Query for the atom.
    7.  Assert that positions and cell match exactly (float precision).
    8.  Assert that custom key-value pairs (metadata) were preserved.
-   **CLI-DB Interaction**:
    -   Although the CLI doesn't do much with the DB yet, we can test a hypothetical `app.py db init` command if implemented, ensuring it creates the `.db` file on disk.

This rigorous testing strategy ensures that the foundation is solid before we start adding complex physics logic in Cycle 02.
