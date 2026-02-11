# Cycle 01 Specification: Core Framework & Configuration

## 1. Summary

The goal of Cycle 01 is to establish the foundational software infrastructure for PyAceMaker (`mlip_autopipec`). This includes setting up the Python package structure, implementing a robust configuration management system using Pydantic, establishing a unified logging framework, and creating the skeleton of the central `Orchestrator` class. By the end of this cycle, the system will be able to parse a complex `config.yaml` file, validate its correctness, initialize the application state, and execute a "Hello World" workflow via a CLI command.

This cycle does **not** involve any scientific computation (DFT, MD, Training). It focuses strictly on software engineering aspects: architecture, data validation, and state management.

## 2. System Architecture

### File Structure

The following file structure will be created. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**                # CLI Entry point (Typer)
├── **constants.py**           # Global constants
├── core/
│   ├── **__init__.py**
│   ├── **orchestrator.py**    # Main workflow controller (Skeleton)
│   ├── **config.py**          # Configuration loading logic
│   ├── **logger.py**          # Logging setup
│   └── **state_manager.py**   # State persistence
├── domain_models/
│   ├── **__init__.py**
│   ├── **config.py**          # Pydantic models for config.yaml
│   ├── **enums.py**           # String Enums (TaskType, etc.)
│   └── **datastructures.py**  # Shared data structures
└── components/                # Placeholders for future cycles
    ├── __init__.py
    ├── base_component.py
    └── mock.py                # Mock implementations for testing
```

### Component Interaction
1.  **User** invokes `mlip-runner run config.yaml`.
2.  **CLI (`main.py`)** reads the file path and calls `load_config`.
3.  **Config Loader (`core/config.py`)** parses YAML and validates it against **Pydantic Models (`domain_models/config.py`)**.
4.  **Orchestrator (`core/orchestrator.py`)** is initialized with the validated config object.
5.  **Logger (`core/logger.py`)** is configured based on the verbosity settings.
6.  **Orchestrator** initializes a "Mock" workflow and logs its progress.

## 3. Design Architecture

### 3.1. Configuration Models (`domain_models/config.py`)
We use Pydantic V2 to enforce strict typing and validation. The configuration is hierarchical.

**Key Models:**

*   `GlobalConfig`: The root model.
    *   `project_name`: str
    *   `orchestrator`: `OrchestratorConfig`
    *   `generator`: `GeneratorConfig`
    *   `oracle`: `OracleConfig`
    *   `trainer`: `TrainerConfig`
    *   `dynamics`: `DynamicsConfig`

*   `OrchestratorConfig`:
    *   `work_dir`: Path (Directory to store results)
    *   `max_iterations`: int (Safety limit)
    *   `state_file`: Path (default: `workflow_state.json`)

*   `ComponentConfig` (Abstract Base):
    *   `enabled`: bool
    *   `type`: StrEnum (e.g., `QE`, `VASP`, `MOCK`)

**Validation Rules:**
-   `work_dir` must be a valid path string.
-   `max_iterations` must be > 0.
-   Enums must match defined `StrEnum` values.

### 3.2. Orchestrator Skeleton (`core/orchestrator.py`)
The `Orchestrator` is a singleton-like class that manages the lifecycle.

```python
class Orchestrator:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.state = WorkflowState()
        self.logger = setup_logging(self.config)

    def run(self):
        self.logger.info("Starting Workflow...")
        # Future cycles will add logic here
        self.logger.info("Workflow Completed.")
```

### 3.3. State Persistence (`core/state_manager.py`)
To ensure recoverability, the system state must be saved to disk after every major step.
-   `WorkflowState`: A Pydantic model tracking `current_iteration`, `completed_tasks`, etc.
-   `StateManager`: Handles loading/saving `workflow_state.json`.

## 4. Implementation Approach

1.  **Setup Project**: Initialize `pyproject.toml` (already done) and create directory structure.
2.  **Define Enums**: Create `domain_models/enums.py` (e.g., `ComponentType`, `TaskStatus`).
3.  **Implement Config Models**: Write `domain_models/config.py`. Add tests to verify validation fails on bad input.
4.  **Implement Logging**: Create `core/logger.py` using standard `logging` library with a custom formatter.
5.  **Implement CLI**: Use `typer` in `main.py` to create the `run` command.
6.  **Implement Orchestrator**: Create the class and wire it up to the CLI.
7.  **Mock Components**: Create dummy classes in `components/mock.py` to satisfy imports if needed.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Validation**:
    -   Test loading a valid YAML file -> Success.
    -   Test loading a YAML with missing required fields -> `ValidationError`.
    -   Test loading a YAML with invalid types (string instead of int) -> `ValidationError`.
-   **Orchestrator Initialization**:
    -   Verify it sets up logger correctly.
    -   Verify it creates the `work_dir` if it doesn't exist.

### 5.2. Integration Testing
-   **CLI Test**:
    -   Run `python -m mlip_autopipec.main --help`.
    -   Run `python -m mlip_autopipec.main run valid_config.yaml` and check if it logs "Workflow Completed".
