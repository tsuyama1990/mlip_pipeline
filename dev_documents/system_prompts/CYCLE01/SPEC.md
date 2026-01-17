# Cycle 01: Core Framework & Configuration

## 1. Summary

Cycle 01 represents the "Genesis" phase of the **MLIP-AutoPipe** project. Its primary objective is to establish the fundamental software infrastructure upon which the entire active learning pipeline will operate. In the context of a "Zero-Human" autonomous system, the robustness of this foundation is paramount. A system that cannot reliably parse its own configuration or persist its state to disk without corruption cannot be trusted to run unsupervised for weeks on an HPC cluster.

This cycle focuses on three critical pillars:
1.  **Strict Configuration Management**: We move away from loose, untyped dictionaries (common in scientific Python scripts) to rigorous, validated **Pydantic** models. This ensures that any input error—whether a typo in a chemical symbol or an unphysical temperature—is caught at startup, not after 48 hours of computation.
2.  **Persistent Data Storage**: We implement a data access layer (DAL) that wraps the `ase.db` (SQLite) functionality. This `DatabaseManager` acts as the single source of truth, ensuring that every atomic structure, energy, and force vector is stored with full provenance metadata.
3.  **Traceable Logging**: We establish a centralized logging architecture that captures every decision made by the system, ensuring that even if the system operates in the dark, it leaves a trail that allows for complete post-mortem reconstruction.

By the end of this cycle, we will have a compiled, tested package structure (`mlip_autopipec`) capable of validating a user's intent and initializing a persistent session, ready for the physics engines to be plugged in.

## 2. System Architecture

The file structure established in this cycle sets the architectural pattern for the entire project. We rigorously separate concerns: configuration logic lives in `config/`, persistent storage logic in `core/`, and the main application entry point is clearly defined.

```ascii
mlip_autopipec/
├── __init__.py           # Package marker exposing version
├── config/               # Sub-package for Data Transfer Objects (DTOs)
│   ├── __init__.py
│   └── models.py         # The Pydantic Schema Definitions (The "Contract")
├── core/                 # Sub-package for Infrastructure
│   ├── __init__.py
│   ├── database.py       # The Repository Implementation (ASE-db wrapper)
│   └── logging.py        # The Observability Layer
└── tests/                # Mirror directory for Unit Tests
    ├── __init__.py
    ├── conftest.py       # Shared Pytest fixtures (e.g., temp DB paths)
    ├── test_config.py    # Validation stress tests
    └── test_database.py  # persistence/retrieval tests
```

### 2.1 Code Blueprints

This section provides the exact specifications for the classes to be implemented. These blueprints serve as the contract for the implementation phase.

#### 2.1.1 Configuration Module (`config/models.py`)

This module defines the "Language" of the system.

**Class `MinimalConfig`**
*   **Purpose**: To capture the user's high-level scientific intent from `input.yaml` without demanding technical minutiae.
*   **Inheritance**: `pydantic.BaseModel`
*   **Configuration**: `model_config = ConfigDict(extra="forbid")` (We reject unknown keys to prevent typo-induced silent errors).
*   **Fields**:
    *   `project_name` (`str`): The unique identifier for the campaign.
        *   *Validation*: Must match regex `^[a-zA-Z0-9_-]+$`. No spaces or special chars to ensure filesystem compatibility.
    *   `elements` (`List[str]`): The chemical system definition.
        *   *Validation*: All items must be valid symbols in `ase.data.chemical_symbols`. List length > 0.
    *   `goal` (`Enum`): A controlled vocabulary of objectives.
        *   *Values*: `melt_quench`, `phase_diagram`, `defect_migration`.
    *   `temperature_range` (`Tuple[float, float]`): The thermodynamic bounds.
        *   *Default*: `(300.0, 1000.0)`.
        *   *Validation*: $T_{min} \ge 0$, $T_{max} \ge T_{min}$.
*   **Factory Method**: `from_yaml(cls, path: FilePath) -> MinimalConfig`
    *   *Logic*: Reads file, parses YAML safely, validates against schema.

**Class `SystemConfig`**
*   **Purpose**: To represent the *execution* state. This includes derived defaults that the user didn't specify but the system needs.
*   **Inheritance**: `pydantic.BaseModel`
*   **Fields**:
    *   `user_config` (`MinimalConfig`): Composition, not inheritance. Preserves the original intent.
    *   `work_dir` (`Path`): Absolute path to the runtime directory.
        *   *Default*: `$CWD/runs/{project_name}`.
    *   `dft_command` (`str`): The executable string for Quantum Espresso.
        *   *Default*: `"mpirun -np {n_cores} pw.x -in {input_file} > {output_file}"`.
    *   `n_cores` (`int`): Parallel width per job.
        *   *Default*: 4.
        *   *Validation*: $> 0$.
    *   `dft_cutoff` (`float`): Plane wave cutoff in Ry.
        *   *Default*: `None` (Deferred to Heuristic Engine).
*   **Factory Method**: `from_minimal(cls, minimal: MinimalConfig) -> SystemConfig`
    *   *Logic*: Instantiates `SystemConfig` by filling in architectural defaults.

#### 2.1.2 Database Module (`core/database.py`)

This module implements the Repository Pattern, abstracting `ase.db`.

**Class `DatabaseManager`**
*   **Purpose**: To provide a thread-safe, schema-enforcing interface to the SQLite database.
*   **Attributes**:
    *   `db_path` (`Path`): The location of the file.
    *   `_lock` (`threading.Lock`): To prevent race conditions during write operations (crucial for Dask threads).
*   **Methods**:
    *   `__init__(self, db_path: Path)`: Sets path, creates parent directories if needed.
    *   `connect(self) -> ase.db.core.Database`: Returns a context-managed connection.
    *   `add_structure(self, atoms: Atoms, metadata: Dict[str, Any] = {}) -> int`:
        *   *Logic*:
            1.  Acquire `_lock`.
            2.  Validate `atoms` is an `ase.Atoms` object.
            3.  Ensure `metadata` contains `config_type` (Schema Enforcement).
            4.  Call `db.write(atoms, **metadata)`.
            5.  Release `_lock`.
            6.  Return the new row ID.
    *   `get_structure(self, id: int) -> Atoms`:
        *   Wraps `db.get(id=id).toatoms()`.
    *   `count(self, selection: str = "") -> int`:
        *   Wraps `db.count(selection)`.
    *   `update_metadata(self, id: int, key: str, value: Any) -> None`:
        *   Updates specific columns for a row (e.g., flagging `trained=True`).

#### 2.1.3 Logging Module (`core/logging.py`)

**Function `setup_logging`**
*   **Purpose**: To configure the global python logging state.
*   **Arguments**: `log_file: Path`, `console_level: int`, `file_level: int`.
*   **Logic**:
    1.  Get root logger `mlip_autopipec`.
    2.  Set propagation to False.
    3.  Create `StreamHandler` (stdout) with simple format: `[INFO] Step completed`.
    4.  Create `FileHandler` (disk) with detailed format: `2023-10-27 10:00:00 | INFO | core.database:45 | Connected`.
    5.  Install `sys.excepthook` handler to catch unhandled crashes and log them with full tracebacks.

### 2.2 Data Flow Mechanics

The data flow in Cycle 01 is linear and initialization-focused.

1.  **Input**: User runs `mlip-auto run inputs.yaml`.
2.  **Parsing**: `config.models` reads the YAML. It checks if "Fe" is a real element. It checks if T=1000K is valid. If fail -> Exit with clear error.
3.  **Expansion**: The `MinimalConfig` is promoted to `SystemConfig`. Defaults are injected.
4.  **Persistence Init**: `core.database` creates `runs/my_proj/mlip.db`. It verifies write permissions.
5.  **Logging Init**: `core.logging` creates `runs/my_proj/system.log`. A "System Start" banner is printed.
6.  **Handover**: The `SystemConfig` object and `DatabaseManager` instance are ready to be passed to the Workflow Manager (Cycle 06).

### 2.3 Design Philosophy

*   **Fail Fast**: We prefer to crash immediately with a `ValidationError` at second 0 rather than producing garbage results at hour 10. Pydantic enforces this.
*   **Dependency Injection**: The `DatabaseManager` will be injected into downstream components (like `DFTRunner`). This makes unit testing easy—we can inject a `MockDatabase` that strictly checks if `add_structure` is called.
*   **No Global State**: Apart from the logger (which is standard practice), we avoid global variables. The `config` object is passed explicitly.

## 3. Design Architecture (Expanded)

### 3.1 Domain Concepts

*   **The Campaign**: Represented by `project_name` + `db`. It is a self-contained unit of work.
*   **The Atom**: The fundamental data unit. It is not just XYZ coordinates; it is XYZ + Energy + Forces + Stress + Provenance.
*   **Provenance**: The concept that every atom knows *how* it was created (`config_type="sqs"`) and *who* calculated it (`calculator="qe"`).

### 3.2 Security Considerations

*   **Path Traversal**: The `project_name` validation prevents users from supplying names like `../../etc/passwd` to overwrite system files.
*   **Command Injection**: While `dft_command` is a string, the system will use `subprocess.run(shell=False)` with argument lists where possible in later cycles to mitigate injection risks.

## 4. Implementation Approach

1.  **Scaffold**: Create the directory tree. Add empty `__init__.py` files.
2.  **Models**: Implement `config/models.py`. Write a small script to try loading a dummy YAML.
3.  **DB**: Implement `core/database.py`. Use TDD: Write a test that tries to write to a locked DB and asserts it waits/retries (or fails gracefully).
4.  **Logging**: Implement `core/logging.py`.
5.  **Integration**: Create a `main.py` prototype that ties them together: Load Config -> Init Log -> Init DB -> Log "Success".

## 5. Test Strategy

### 5.1 Unit Testing (`tests/test_config.py`, `tests/test_database.py`)

*   **Config Validation**:
    *   *Input*: YAML with `elements: ["Unobtainium"]`.
    *   *Expectation*: `ValidationError` citing invalid element.
    *   *Input*: YAML with `temperature_range: [1000, 300]` (min > max).
    *   *Expectation*: `ValidationError`.
*   **Database**:
    *   *Scenario*: Concurrency.
    *   *Setup*: Spawn 10 threads that all try to write to the DB using the same `DatabaseManager`.
    *   *Expectation*: All 10 writes succeed (due to internal locking), `count()` returns 10.
    *   *Scenario*: Metadata.
    *   *Action*: Write atom with `info={'tag': 'test'}`. Retrieve it.
    *   *Expectation*: `atom.info['tag'] == 'test'`.

### 5.2 Integration Testing

*   **Startup Sequence**:
    *   Create a valid `input.yaml`.
    *   Run a script that imports the modules and mimics the startup sequence.
    *   Check if `runs/project/mlip.db` exists.
    *   Check if `runs/project/system.log` exists and contains "System initialized".
