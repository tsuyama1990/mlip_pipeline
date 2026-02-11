# Cycle 01 Specification: Core Framework & Orchestrator Skeleton

## 1. Summary
This cycle establishes the foundation of the PYACEMAKER system. The primary goal is to create the "spinal cord" of the application: the CLI entry point, the configuration parser, the state management system, and the central `Orchestrator` loop. At this stage, the orchestrator will not perform real scientific calculations (DFT/MD) but will successfully transition through the lifecycle states using "Mock" components. This ensures that the control flow, directory management, and error handling mechanisms are robust before complex physics modules are integrated.

## 2. System Architecture

### 2.1. File Structure
The following file structure will be implemented in this cycle.

```text
src/mlip_autopipec/
├── __init__.py
├── main.py                     # [CREATE] CLI Entry point (Typer)
├── constants.py                # [CREATE] System constants
├── core/
│   ├── __init__.py
│   ├── orchestrator.py         # [CREATE] The main control loop
│   ├── state_manager.py        # [CREATE] JSON state persistence
│   └── logger.py               # [CREATE] Logging configuration
├── domain_models/
│   ├── __init__.py
│   ├── config.py               # [CREATE] Pydantic models for config.yaml
│   ├── datastructures.py       # [CREATE] WorkflowState model
│   └── enums.py                # [CREATE] StrEnums for system states
└── utils/
    ├── __init__.py
    └── io.py                   # [CREATE] YAML/JSON helpers
```

### 2.2. Component Interaction
1.  **User** runs `mlip-runner run config.yaml`.
2.  **`main.py`** parses arguments and calls `Orchestrator`.
3.  **`Orchestrator`**:
    *   Loads `config.yaml` into `GlobalConfig` (validated).
    *   Initializes `StateManager` to check for existing `.state` file.
    *   Sets up `Logger`.
    *   Enters the `run()` loop.
    *   Iterates through 5 cycles (default).
    *   In each cycle, calls placeholder methods (`explore()`, `label()`, `train()`, `validate()`).
    *   Updates `WorkflowState` after each step.
4.  **`StateManager`** saves `workflow_state.json` atomically to disk.

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)

#### `enums.py`
Use `StrEnum` (Python 3.11+) for type safety.
*   `WorkflowStatus`: `IDLE`, `EXPLORATION`, `LABELING`, `TRAINING`, `POST_PROCESSING`, `COMPLETED`, `FAILED`.

#### `config.py`
Strict Pydantic models.
*   `OrchestratorConfig`: `work_dir` (Path), `max_cycles` (int).
*   `GlobalConfig`: The root model containing `orchestrator`, `dft`, `training`, etc. (Components can be optional/dict for now).

#### `datastructures.py`
*   `WorkflowState`:
    *   `iteration`: int
    *   `status`: WorkflowStatus
    *   `current_potential_path`: Path | None
    *   `current_dataset_path`: Path | None

### 3.2. Core Logic (`core/`)

#### `state_manager.py`
*   **Responsibility**: Load/Save `WorkflowState`.
*   **Constraint**: Must use atomic writes (write to temp, then rename) to prevent corruption if the process is killed during save.

#### `orchestrator.py`
*   **Responsibility**: The state machine.
*   **Logic**:
    ```python
    while self.state.iteration < self.config.max_cycles:
        self.explore() # Log "Exploring..."
        self.label()   # Log "Labeling..."
        self.train()   # Log "Training..."
        self.state.iteration += 1
        self.save_state()
    ```

## 4. Implementation Approach

### Step 1: Domain Models
Define the Pydantic models first. This acts as the contract for the rest of the system.
*   Create `enums.py` with `WorkflowStatus`.
*   Create `config.py` with basic `GlobalConfig`.
*   Create `datastructures.py` for state tracking.

### Step 2: Utilities & Logging
*   Implement `utils/io.py` (safe YAML loading).
*   Implement `core/logger.py` using standard `logging` library but with a custom formatter and file handler (writing to `work_dir/mlip.log`).

### Step 3: State Manager
*   Implement `StateManager` class.
*   Add unit tests to verify atomic writes and state recovery.

### Step 4: Orchestrator (Mock Implementation)
*   Implement the `Orchestrator` class.
*   Inject dependencies (Config, StateManager).
*   Write the main loop with logging statements acting as placeholders for future modules.

### Step 5: CLI Entry Point
*   Use `typer` to create the command line interface.
*   Command: `mlip-runner run <config_path>`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_config.py`**:
    *   Load a valid YAML -> Assert success.
    *   Load an invalid YAML (missing field) -> Assert `ValidationError`.
    *   Test path validation (e.g., `work_dir` creation).
*   **`test_state_manager.py`**:
    *   Save state, load state, assert equality.
    *   Test resilience against partial writes (mocking interrupt).

### 5.2. Integration Testing
*   **`test_orchestrator_flow.py`**:
    *   Run the orchestrator with `max_cycles=2`.
    *   Check if `workflow_state.json` shows `iteration=2` and `status=COMPLETED`.
    *   Check if `mlip.log` contains expected sequence of messages.
