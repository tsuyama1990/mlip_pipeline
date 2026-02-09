# Cycle 01 Specification: Core Framework & Configuration

## 1. Summary

The objective of Cycle 01 is to establish the fundamental "skeleton" of the PYACEMAKER system. Before implementing complex physics logic, we must ensure a robust software engineering foundation. This cycle focuses on:
1.  **Project Scaffolding**: Creating the directory structure and entry points.
2.  **Configuration Management**: Implementing strict Pydantic models to validate the user's `config.yaml`. This enforces the "Zero-Config" philosophy by catching errors early (fail-fast).
3.  **Domain Modeling**: Defining the core data structures (`Structure`, `Potential`) that will be passed between components.
4.  **State Management**: Implementing a `StateManager` to handle the file system operations (creating `active_learning/` directories) and ensure the system can resume from interruptions.
5.  **Abstract Base Classes (ABCs)**: Defining the interfaces for the Generator, Oracle, Trainer, and Dynamics components to ensure modularity.

By the end of this cycle, we will have a runnable `main.py` that can read a configuration file, validate it, set up the workspace, and log its status, even though the actual physics engines are not yet connected.

## 2. System Architecture

This cycle builds the central `Orchestrator` and the data definitions.

### File Structure
Files to be created in this cycle are marked in **bold**.

```
mlip-pipeline/
├── pyproject.toml
├── src/
│   └── mlip_autopipec/
│       ├── **__init__.py**
│       ├── **main.py**                     # Entry point (CLI)
│       ├── **constants.py**                # Global constants
│       ├── **factory.py**                  # Component Factory
│       ├── **logging.py**                  # Logging configuration
│       ├── core/
│       │   ├── **__init__.py**
│       │   ├── **orchestrator.py**         # Main application logic (Skeleton)
│       │   ├── **state_manager.py**        # File system & Checkpointing
│       │   └── **exceptions.py**           # Custom exceptions
│       ├── domain_models/
│       │   ├── **__init__.py**
│       │   ├── **config.py**               # Config Pydantic Models
│       │   ├── **structure.py**            # Structure Data Model
│       │   ├── **potential.py**            # Potential Data Model
│       │   └── **enums.py**                # String Enums
│       └── components/
│           ├── **__init__.py**
│           ├── generator/
│           │   ├── **__init__.py**
│           │   └── **base.py**             # Abstract Base Class
│           ├── oracle/
│           │   ├── **__init__.py**
│           │   └── **base.py**             # Abstract Base Class
│           ├── trainer/
│           │   ├── **__init__.py**
│           │   └── **base.py**             # Abstract Base Class
│           ├── dynamics/
│           │   ├── **__init__.py**
│           │   └── **base.py**             # Abstract Base Class
│           └── validator/
│               ├── **__init__.py**
│               └── **base.py**             # Abstract Base Class
└── tests/
    ├── **test_config.py**
    └── **test_orchestrator.py**
```

## 3. Design Architecture

We use **Pydantic V2** for all domain models to ensure strict type safety and serialization.

### 3.1. Configuration Models (`domain_models/config.py`)
The configuration is a hierarchical structure.
*   `Config`: Root model.
    *   `structure_generation`: `GeneratorConfig`
    *   `dft_computation`: `OracleConfig`
    *   `potential_training`: `TrainerConfig`
    *   `active_learning`: `DynamicsConfig`
*   **Constraints**:
    *   `work_dir`: Must be a valid path string.
    *   `iterations`: Must be > 0.
    *   Validation: All paths must be checked (though creation happens in `StateManager`).

### 3.2. Domain Objects (`domain_models/structure.py`, `potential.py`)
*   `Structure`: A wrapper around `ase.Atoms`.
    *   Attributes: `atoms` (ASE object), `provenance` (StringEnum: "random", "md_halt", "defect"), `features` (dict), `labels` (Energy/Forces/Stress).
    *   **Invariant**: The `atoms` object must be serializable (e.g., to JSON/Pickle).
*   `Potential`:
    *   Attributes: `id` (UUID), `path` (Path), `parent_id` (UUID), `baseline_config` (Dict).

### 3.3. State Manager (`core/state_manager.py`)
*   Responsibilities:
    *   `initialize_workspace()`: Creates `active_learning/`, `data/`, `logs/`.
    *   `create_cycle_dir(cycle_id)`: Creates `active_learning/cycle_001`.
    *   `save_state(state)`: Serializes the current progress (cycle number) to `state.json`.
    *   `load_state()`: Recovers execution from a crash.

## 4. Implementation Approach

1.  **Setup Environment**: Initialize `src/mlip_autopipec` package.
2.  **Define Enums**: Create `GeneratorType`, `OracleType`, etc., in `enums.py`.
3.  **Implement Config Models**: Write the Pydantic models in `config.py`. Add validators (e.g., check if `potential_path` exists if specified).
4.  **Implement Domain Models**: Create `Structure` and `Potential` classes. Ensure `Structure` can ingest an `ase.Atoms` object.
5.  **Implement Core Utilities**:
    *   `logging.py`: Setup `structlog` or standard `logging` with JSON formatter for machine readability.
    *   `state_manager.py`: Implement directory creation logic.
6.  **Define Interfaces (ABCs)**: In `components/*/base.py`, define the abstract methods (e.g., `BaseOracle.compute(structures) -> List[Structure]`).
7.  **Implement Orchestrator Skeleton**:
    *   `__init__(config_path)`: Load config, init StateManager.
    *   `run()`: A simple loop that calls `state_manager.create_cycle_dir()` and logs "Starting cycle X".
8.  **CLI Entry Point**: `main.py` using `argparse` to accept `--config`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Validation**:
    *   Create `tests/test_config.py`.
    *   Test Case: Load a valid YAML -> Assert Pydantic model is created.
    *   Test Case: Load YAML with missing fields -> Assert `ValidationError`.
    *   Test Case: Load YAML with invalid types (string instead of int) -> Assert `ValidationError`.
*   **State Management**:
    *   Test Case: `initialize_workspace` creates folders. Use `tmp_path` fixture.
    *   Test Case: `save_state` writes JSON. `load_state` reads it back.

### 5.2. Integration Testing
*   **Orchestrator Run**:
    *   Create a minimal `config_test.yaml`.
    *   Run `Orchestrator(config).run()`.
    *   Assert that the directory tree `active_learning/cycle_001` is physically created.
    *   Assert that `main.log` contains expected log messages.
