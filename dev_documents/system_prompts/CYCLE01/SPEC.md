# Cycle 01 Specification: Foundation

## 1. Summary

Cycle 01 lays the groundwork for the PYACEMAKER project. The primary objective is to establish the project structure, configuration management, logging infrastructure, and the core domain models that will be used throughout the system. This cycle delivers the CLI entry point (`mlip-auto`) and the state management logic (`WorkflowState`), ensuring that subsequent cycles have a robust framework to build upon. No scientific calculation logic is implemented here, but the interfaces (protocols) for them are defined.

## 2. System Architecture

The focus of this cycle is on the **Infrastructure** and **Domain Models** layers, and the **Orchestration** skeleton.

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **app.py**                        # Main CLI entry point (Typer)
├── **constants.py**                  # Project-wide constants
├── **cli/**
│   ├── **__init__.py**
│   └── **commands.py**               # CLI command implementations (init, run-loop)
├── **domain_models/**
│   ├── **__init__.py**
│   ├── **config.py**                 # Global Config Pydantic models
│   ├── **structure.py**              # Structure & Candidate models
│   └── **workflow.py**               # WorkflowState models
├── **infrastructure/**
│   ├── **__init__.py**
│   ├── **logging.py**                # Rich-based logging setup
│   └── **io.py**                     # YAML/Pickle I/O utilities
└── **orchestration/**
    ├── **__init__.py**
    └── **workflow.py**               # WorkflowManager skeleton
```

## 3. Design Architecture

### Domain Models (`domain_models/`)

#### `config.py`
Defines the `Config` schema using Pydantic. It must be strictly typed and support loading from YAML.
*   **`Config`**: Root model containing sub-configs for `structure_gen`, `oracle`, `trainer`, `dynamics`, and `orchestrator`.
*   **Validation**: Ensure paths (if provided) are valid or creatable. Default values should be provided for "Zero-Config" usage where possible.

#### `structure.py`
Defines the atomic structure representation used internally.
*   **`Structure`**: Wrapper around `ase.Atoms` (or a serializable equivalent). Must support serialization to/from JSON/YAML.
*   **`Candidate`**: Extends `Structure` with metadata (source, priority, status).

#### `workflow.py`
Defines the state of the active learning loop.
*   **`WorkflowState`**:
    *   `cycle_index`: int
    *   `current_phase`: Enum (EXPLORATION, ORACLE, TRAINING, etc.)
    *   `dataset_stats`: Dict
    *   `is_halted`: bool

### Infrastructure (`infrastructure/`)

#### `logging.py`
*   Uses `rich` library for colored console output.
*   Configures a root logger that writes to both console (INFO) and a file (DEBUG).
*   Format: `[Timestamp] [Level] [Module] Message`.

#### `io.py`
*   **`load_yaml(path) -> Dict`**: Safe YAML loading.
*   **`dump_yaml(data, path)`**: YAML dumping.
*   **`save_state(state, path)`**: Persist `WorkflowState`.
*   **`load_state(path) -> WorkflowState`**: Load `WorkflowState`.

### CLI (`app.py`, `cli/commands.py`)
*   Uses `typer` for command-line interface.
*   **`init`**: Generates a template `config.yaml` in the current directory.
*   **`run-loop`**: The main entry point. Loads config, initializes `WorkflowManager`, and starts the loop (mocked for now).

## 4. Implementation Approach

1.  **Setup Project**: Initialize `src/mlip_autopipec` package structure.
2.  **Implement Constants**: Define defaults in `constants.py`.
3.  **Implement Infrastructure**: Create `logging.py` and `io.py`.
4.  **Implement Domain Models**:
    *   Create `structure.py` (basic `Candidate` model).
    *   Create `workflow.py` (`WorkflowState`).
    *   Create `config.py` (Hierarchical configuration).
5.  **Implement CLI**:
    *   Create `app.py` and `cli/commands.py`.
    *   Implement `init` command to write default config.
    *   Implement `run-loop` command to load config and log "Starting loop".
6.  **Implement Workflow Skeleton**:
    *   Create `WorkflowManager` in `orchestration/workflow.py`.
    *   Implement `load_state` and `save_state` methods using `io.py`.

## 5. Test Strategy

### Unit Testing
*   **`test_config.py`**:
    *   Load a valid YAML and verify `Config` object creation.
    *   Test missing required fields raises `ValidationError`.
*   **`test_workflow_state.py`**:
    *   Create a `WorkflowState`, save it to a temp file, load it back, and assert equality.
*   **`test_io.py`**:
    *   Test YAML read/write with mock data.

### Integration Testing
*   **`test_cli.py`**:
    *   Use `typer.testing.CliRunner`.
    *   Run `mlip-auto init` and verify `config.yaml` is created.
    *   Run `mlip-auto run-loop` with the created config and verify log output.
