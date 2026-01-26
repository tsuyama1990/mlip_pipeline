# Cycle 01: Skeleton & Configuration

## 1. Summary
This cycle establishes the foundational infrastructure of **PyAcemaker**. We will set up the project directory structure, configure the build system, implement the Command Line Interface (CLI), and define the robust configuration management system using **Pydantic**. This cycle ensures that the system can parse user inputs (`input.yaml`), initialize the logging system, and maintain a valid internal state, paving the way for the functional modules in subsequent cycles.

## 2. System Architecture

We will create the core package structure `mlip_autopipec` and the essential orchestration components.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
.
├── pyproject.toml
├── README.md
├── **src/**
│   └── **mlip_autopipec/**
│       ├── **__init__.py**
│       ├── **app.py**               # Entry point for CLI (Typer)
│       ├── **config/**
│       │   ├── **__init__.py**
│       │   ├── **models.py**        # Pydantic schemas aggregation
│       │   └── **schemas/**         # Modular configuration schemas
│       │       ├── **workflow.py**
│       │       ├── **dft.py**
│       │       └── ...
│       ├── **data_models/**         # Domain Models
│       │   ├── **__init__.py**
│       │   ├── **state.py**         # Workflow State model
│       │   └── **candidate.py**     # Candidate structure model
│       ├── **modules/**
│       │   └── **cli_handlers/**    # CLI Logic Handlers
│       │       └── **handlers.py**
│       └── **orchestration/**
│           ├── **__init__.py**
│           └── **workflow.py**      # Workflow Manager
└── **tests/**
    ├── **conftest.py**
    ├── **unit/**
    │   ├── **test_config.py**
    │   └── **test_cli_validation.py**
    └── **e2e/**
        └── **test_cycle01.py**
```

## 3. Design Architecture

### Configuration Management (Schema-First Design)
We will use **Pydantic** to define the configuration schema. This ensures that `input.yaml` is validated strictly at startup.

**Key Models (`src/mlip_autopipec/config/models.py` & `schemas/`):**
*   `WorkflowConfig`: The root configuration object (in `schemas/workflow.py`).
*   `DFTConfig`: Parameters for Quantum Espresso (in `schemas/dft.py`).
*   `TrainingConfig`: Parameters for Pacemaker.
*   `ValidationConfig`: Thresholds for validation metrics.

**Key Invariants:**
*   All paths (e.g., potential files, executables) must be validated to exist (if input) or be writable (if output).
*   Numerical parameters (e.g., temperatures) must be within physical ranges (checked via Pydantic validators).

### Domain Models
*   `WorkflowState` (in `data_models/state.py`): Tracks the current cycle number (0-based) and status.

### CLI & Orchestration
*   **CLI**: Uses `typer` in `app.py` to handle commands.
    *   `mlip-auto init`: Generates a default `input.yaml`.
    *   `mlip-auto run loop` (or `mlip-auto run`): Loads config and initializes Workflow Manager.
    *   `mlip-auto validate`: Validates config or runs physics checks.
*   **Workflow Manager**: The central class initialized with `WorkflowConfig`. In this cycle, it will simply load the config and log the startup process.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `src/mlip_autopipec` and `tests`.
2.  **Configuration Schemas**: Implement `WorkflowConfig` and sub-models in `config/schemas/`. Aggregate in `config/models.py`.
3.  **State Management**: Implement `WorkflowState` in `data_models/state.py` to track the current cycle number and status.
4.  **Logging**: Implement centralized logging in `utils/logging.py`.
5.  **CLI Implementation**: Create `app.py` and `modules/cli_handlers/handlers.py`.
    *   `mlip-auto init`: Generates a default `input.yaml`.
    *   `mlip-auto run`: Starts the workflow loop.
6.  **Orchestrator Stub**: Create `WorkflowManager` in `orchestration/workflow.py`.

## 5. Test Strategy

### Unit Testing
*   **`test_config.py`**:
    *   Test loading a valid `input.yaml`.
    *   Test validation errors (e.g., negative temperature, missing fields).
    *   Test default value injection.
*   **`test_cli_validation.py`**:
    *   Verify `mlip-auto --help` works.
    *   Verify `mlip-auto init` creates a file.
    *   Verify `mlip-auto validate` checks config.

### Integration Testing
*   **Config-to-Orchestrator Handshake**: Verify that the `WorkflowManager` correctly receives the `WorkflowConfig` object.
