# Cycle 01 Specification: Core Infrastructure & Domain Models

## 1. Summary
**Goal**: Establish the foundational infrastructure for the PYACEMAKER system. This cycle focuses on defining the "Language" of the system (Pydantic Domain Models), implementing robust configuration management, centralised logging, and the basic CLI entry point.

**Key Features**:
*   Strict Pydantic V2 schemas for all core data structures (Atoms, Potentials, Config).
*   YAML configuration parser with environment variable substitution.
*   Atomic State Manager to track workflow progress and ensure resume capability.
*   Application scaffold (CLI) using `typer`.

## 2. System Architecture

The following file structure will be created. **Bold** files are to be implemented in this cycle.

```ascii
project_root/
├── **config.yaml**                 # Example Configuration
├── **src/**
│   └── **mlip_autopipec/**
│       ├── **__init__.py**
│       ├── **constants.py**        # Global Constants
│       ├── **main.py**             # CLI Entry Point
│       ├── **core/**
│       │   ├── **__init__.py**
│       │   ├── **config_parser.py** # YAML Loader
│       │   ├── **state_manager.py** # Workflow State Tracker
│       │   └── **logger.py**       # Centralised Logging
│       └── **domain_models/**
│           ├── **__init__.py**
│           ├── **config.py**       # Config Schemas
│           ├── **datastructures.py** # Domain Entities
│           └── **enums.py**        # Enumerations
└── **tests/**
    ├── **__init__.py**
    ├── **conftest.py**
    ├── **test_core/**
    │   ├── **test_config.py**
    │   └── **test_state.py**
    └── **test_domain/**
        └── **test_models.py**
```

## 3. Design Architecture

This cycle adopts a **Schema-First Design**. All business logic depends on these schemas.

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/`)

#### `enums.py`
*   **`TaskStatus`**: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`.
*   **`ComponentRole`**: `GENERATOR`, `ORACLE`, `TRAINER`, `DYNAMICS`, `VALIDATOR`.

#### `datastructures.py`
*   **`Structure`**:
    *   Wraps `ase.Atoms` but adds provenance metadata.
    *   Fields: `uid` (UUID), `atoms` (ASE object), `provenance` (str), `tags` (dict), `energy` (Optional[float]), `forces` (Optional[Array]).
    *   **Invariant**: `len(forces) == len(atoms)` if forces are present.
*   **`WorkflowState`**:
    *   Tracks the global state of the active learning loop.
    *   Fields: `iteration` (int), `current_potential_path` (Path), `dataset_path` (Path), `status` (TaskStatus).

#### `config.py`
*   **`OrchestratorConfig`**: `work_dir`, `max_iterations`.
*   **`GeneratorConfig`**: Discriminator union for different strategies (Random, M3GNet).
*   **`OracleConfig`**: DFT settings (command, mixing_beta).
*   **`TrainerConfig`**: Pacemaker settings (r_cut, max_deg).
*   **`FullConfig`**: The root model aggregating all above.

### 3.2. Core Infrastructure (`src/mlip_autopipec/core/`)

#### `config_parser.py`
*   Loads YAML files.
*   Performs Pydantic validation.
*   **Feature**: Allows `${VAR}` syntax for environment variable substitution (e.g., for API keys or paths).

#### `state_manager.py`
*   **Responsibility**: Persistence of `WorkflowState`.
*   **Mechanism**: Atomic writes to `workflow_state.json` to prevent corruption during crashes.
*   **Method**: `load_state()`, `save_state(state)`.

#### `main.py`
*   Uses `typer` to define commands:
    *   `mlip-runner run config.yaml`: Starts the orchestration.
    *   `mlip-runner init`: Generates a default config file.

## 4. Implementation Approach

1.  **Define Enums & Data Structures**: Start with `domain_models/enums.py` and `datastructures.py`. Ensure strict type checking.
2.  **Define Configuration Schema**: Create `domain_models/config.py`.
3.  **Implement Core Utils**: Write `logger.py` and `constants.py`.
4.  **Implement Config Parser**: Write `core/config_parser.py` with tests for validation and env var substitution.
5.  **Implement State Manager**: Write `core/state_manager.py` with atomic write logic.
6.  **CLI Skeleton**: Create `main.py` wiring up the `init` command (which dumps a default config schema).

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_models.py`**:
    *   Verify `Structure` raises validation error if arrays have mismatched shapes.
    *   Verify `FullConfig` rejects invalid YAML structures.
*   **`test_config.py`**:
    *   Test env var substitution: `work_dir: ${HOME}/mlip_runs`.
*   **`test_state.py`**:
    *   Test atomic writes: Interrupt a save operation and ensure the file is not corrupted (mocking).

### 5.2. Integration Testing
*   **CLI Test**: Run `mlip-runner init` and assert `config.yaml` is created and valid.
