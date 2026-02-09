# Cycle 01 Specification: Core Framework

## 1. Summary

This cycle establishes the **Core Framework** of the PyAceMaker system. It includes the `Orchestrator`, the central nervous system that manages the active learning loop, configuration management, logging, and the abstract base classes for all future components (Generator, Oracle, Trainer, Dynamics, Validator).

## 2. System Architecture

The following file structure will be created.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**               # CLI Entry Point
├── **constants.py**          # System Constants
├── core/
│   ├── **__init__.py**
│   ├── **orchestrator.py**   # Main Logic
│   └── **logger.py**         # Logging Setup
├── domain_models/
│   ├── **__init__.py**
│   ├── **config.py**         # Pydantic Configuration Models
│   └── **enums.py**          # Component Roles & Types
└── components/
    ├── **__init__.py**
    └── **base.py**           # Abstract Base Classes
```

## 3. Design Architecture

### 3.1. Configuration (Schema-First)
We use `pydantic` to define strict schemas for `config.yaml`.
*   **Location**: `src/mlip_autopipec/domain_models/config.py`
*   **Models**:
    *   `OrchestratorConfig`: `work_dir`, `max_cycles`, `uncertainty_threshold`.
    *   `BaseComponentConfig`: Common fields for all components.
    *   `Config`: The root model containing `orchestrator` and component configs.

### 3.2. The Orchestrator
*   **Location**: `src/mlip_autopipec/core/orchestrator.py`
*   **Responsibility**:
    1.  Load and validate `config.yaml`.
    2.  Initialize the directory structure (`active_learning/`, `potentials/`, etc.).
    3.  Instantiate components based on config.
    4.  Manage the `run_cycle` loop (although logic is minimal in Cycle 01, the structure must be there).

### 3.3. Logging
*   **Location**: `src/mlip_autopipec/core/logger.py`
*   **Features**:
    *   Console logging (INFO level).
    *   File logging (DEBUG level) in `orchestrator.log`.

### 3.4. Base Components
*   **Location**: `src/mlip_autopipec/components/base.py`
*   **Classes**:
    *   `BaseComponent`: Inherits `ABC`.
    *   `BaseGenerator`, `BaseOracle`, `BaseTrainer`, `BaseDynamics`, `BaseValidator`: Abstract classes defining the interface `explore()`, `compute()`, `train()`, etc.

## 4. Implementation Approach

1.  **Define Models**: Implement `domain_models/config.py` first.
2.  **Setup Logging**: Implement `core/logger.py`.
3.  **Base Classes**: Implement `components/base.py` to allow type hinting in Orchestrator.
4.  **Orchestrator**: Implement `core/orchestrator.py` which loads config and sets up the environment.
5.  **CLI**: Implement `main.py` to trigger the Orchestrator.

## 5. Constraints
*   Use `pydantic` V2.
*   Strict typing (no `Any` if possible).
*   Follow strict linting rules (`ruff`, `mypy`).
