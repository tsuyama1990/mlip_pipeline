# Cycle 01: Skeleton & Configuration

## 1. Summary
This cycle establishes the foundational infrastructure of **PyAcemaker**. We will set up the project directory structure, configure the build system, implement the Command Line Interface (CLI), and define the robust configuration management system using **Pydantic**. This cycle ensures that the system can parse user inputs (`config.yaml`), initialize the logging system, and maintain a valid internal state, paving the way for the functional modules in subsequent cycles.

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
│       ├── **main.py**              # Entry point for CLI
│       ├── **cli.py**               # CLI argument parsing (Typer/Click)
│       ├── **logging_config.py**    # Centralized logging setup
│       ├── **config/**
│       │   ├── **__init__.py**
│       │   ├── **models.py**        # Pydantic schemas for Config
│       │   └── **defaults.py**      # Default configuration values
│       └── **orchestration/**
│           ├── **__init__.py**
│           ├── **coordinator.py**   # Main Orchestrator class (Stub)
│           └── **state.py**         # Workflow State model
└── **tests/**
    ├── **conftest.py**
    ├── **test_config.py**
    └── **test_cli.py**
```

## 3. Design Architecture

### Configuration Management (Schema-First Design)
We will use **Pydantic** to define the configuration schema. This ensures that `config.yaml` is validated strictly at startup, preventing runtime errors due to missing or invalid parameters.

**Key Models (`src/mlip_autopipec/config/models.py`):**
*   `WorkflowConfig`: The root configuration object.
*   `DFTConfig`: Parameters for Quantum Espresso (e.g., command, k-spacing).
*   `TrainingConfig`: Parameters for Pacemaker (e.g., cutoff, basis size).
*   `DynamicsConfig`: Parameters for LAMMPS (e.g., temperature, steps).
*   `ValidationConfig`: Thresholds for validation metrics.

**Key Invariants:**
*   All paths (e.g., potential files, executables) must be validated to exist (if input) or be writable (if output).
*   Numerical parameters (e.g., temperatures) must be within physical ranges (checked via Pydantic validators).

### CLI & Orchestration
*   **CLI**: Uses `argparse` or `typer` to handle commands like `init` (create sample config) and `run` (start workflow).
*   **Orchestrator**: The central class initialized with `WorkflowConfig`. In this cycle, it will simply load the config and log the startup process.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `src/mlip_autopipec` and `tests`.
2.  **Configuration Schemas**: Implement `WorkflowConfig` and sub-models in `config/models.py`. Add unit tests to verify valid/invalid YAML parsing.
3.  **State Management**: Implement `WorkflowState` in `orchestration/state.py` to track the current cycle number and status.
4.  **Logging**: Implement `logging_config.py` to output formatted logs to both console (INFO) and file (DEBUG).
5.  **CLI Implementation**: Create `main.py` and `cli.py` to expose the entry point.
    *   `mlip-auto init`: Generates a default `config.yaml`.
    *   `mlip-auto run --config config.yaml`: Loads config and initializes Orchestrator.
6.  **Orchestrator Stub**: Create the `Orchestrator` class in `orchestration/coordinator.py` that accepts the config and initializes the state.

## 5. Test Strategy

### Unit Testing
*   **`test_config.py`**:
    *   Test loading a valid `config.yaml`.
    *   Test validation errors (e.g., negative temperature, missing fields).
    *   Test default value injection.
*   **`test_cli.py`**:
    *   Verify `mlip-auto --help` works.
    *   Verify `mlip-auto init` creates a file.
    *   Verify `mlip-auto run` fails gracefully without a config file.

### Integration Testing
*   **Config-to-Orchestrator Handshake**: Verify that the `Orchestrator` correctly receives the `WorkflowConfig` object instantiated by the CLI.
