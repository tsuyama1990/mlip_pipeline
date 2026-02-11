# Cycle 01 Specification: Core Framework & Infrastructure

## 1. Summary

Cycle 01 focuses on establishing the foundational infrastructure of **PyAceMaker**. The primary goal is to create a robust, type-safe, and modular skeleton that can support the complex active learning workflows developed in subsequent cycles. This cycle avoids implementing heavy physics logic (like DFT or MD integration) and instead focuses on the "Business Logic" of the orchestration, configuration management, and logging.

Crucially, this cycle implements **Mock Components**. These are lightweight, dependency-free implementations of the core interfaces (Oracle, Trainer, etc.) that simulate the behavior of external engines. This allows us to verify the orchestrator's state management and error handling logic immediately, without needing a full HPC environment or external binaries installed.

## 2. System Architecture

The file structure is designed to separate domain models (data) from business logic (components) and infrastructure (core).

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── **pyproject.toml**             # Dependencies (pydantic, typer, PyYAML)
├── **src/**
│   └── **mlip_autopipec/**
│       ├── **__init__.py**
│       ├── **main.py**            # CLI Entry Point (Typer)
│       ├── **domain_models/**     # Pydantic Schemas
│       │   ├── **__init__.py**
│       │   ├── **config.py**      # Config, OrchestratorConfig, ComponentConfigs
│       │   ├── **inputs.py**      # Structure, Job, ProjectState
│       │   ├── **results.py**     # CalculationResult, TrainingResult
│       │   └── **enums.py**       # StrEnum definitions (ComponentRole, Status)
│       ├── **core/**
│       │   ├── **__init__.py**
│       │   ├── **orchestrator.py** # The "Brain" class
│       │   ├── **state_manager.py** # State persistence (JSON)
│       │   ├── **logger.py**       # Logging setup
│       │   └── **exceptions.py**   # Custom exceptions
│       └── **components/**
│           ├── **__init__.py**
│           ├── **base.py**         # Abstract Base Classes (ABC)
│           └── **mock.py**         # Mock implementations for Cycle 01
└── **tests/**
    ├── **conftest.py**
    ├── **test_config.py**
    ├── **test_orchestrator.py**
    └── **test_state_manager.py**
```

### Key Components
1.  **Domain Models (`src/mlip_autopipec/domain_models/`)**: The single source of truth for data structures. We use Pydantic V2 for validation.
2.  **Orchestrator (`src/mlip_autopipec/core/orchestrator.py`)**: The main controller. In this cycle, it will initialize components based on the config and run a "Mock Loop".
3.  **State Manager (`src/mlip_autopipec/core/state_manager.py`)**: Handles saving and loading the `workflow_state.json` to ensure the process can be paused and resumed.
4.  **Mock Components (`src/mlip_autopipec/components/mock.py`)**: Implementations of `BaseGenerator`, `BaseOracle`, etc., that return dummy data (e.g., random energies) to allow the loop to run.

## 3. Design Architecture

This system relies heavily on **Pydantic V2** to enforce type safety and configuration validity.

### 3.1. Domain Models
*   **Enums (`enums.py`)**: Define `ComponentRole` (Generator, Oracle, etc.) and `TaskStatus` (PENDING, RUNNING, COMPLETED) to avoid magic strings.
*   **Configuration (`config.py`)**:
    *   `GlobalConfig`: Root model containing sections for `orchestrator`, `generator`, `oracle`, etc.
    *   **Discriminated Unions**: Use a `type` field to distinguish between implementations. For example, `GeneratorConfig` is a Union of `RandomGeneratorConfig` and `MockGeneratorConfig`.
*   **Data Structures (`inputs.py`)**:
    *   `Structure`: A simplified representation of an atomic structure (positions, numbers, cell, pbc) that can be converted to/from ASE Atoms. Includes a `tags` dict for metadata.
    *   `ProjectState`: Tracks the current iteration, status, and paths to current potential/dataset.

### 3.2. Orchestrator Logic
The `Orchestrator` class follows the **State Pattern**.
*   **Inputs**: A `Config` object and a working directory path.
*   **Responsibilities**:
    1.  Validate the configuration.
    2.  Initialize the `StateManager`.
    3.  Instantiate component classes (Generator, Oracle...) based on the config.
    4.  Execute the main loop: `Generate -> Label -> Train -> Verify`.
*   **Error Handling**: Catches specific `PyAceError` exceptions and logs them before safely shutting down.

### 3.3. Logging
*   Centralized logging configuration in `logger.py`.
*   Uses `structlog` or standard `logging` with JSON formatter to make logs machine-parsable.
*   Console output should be human-readable.

## 4. Implementation Approach

1.  **Environment Setup**: Initialize `uv` project and add dependencies: `pydantic`, `typer`, `pyyaml`, `ase` (for structure handling), `numpy`.
2.  **Domain Modeling**: Implement `enums.py`, then `inputs.py`, and finally `config.py`. Write tests to ensure invalid configs raise `ValidationError`.
3.  **Core Utilities**: Implement `logger.py` and `exceptions.py`.
4.  **Base Components**: Define the abstract base classes in `components/base.py` (e.g., `BaseGenerator` with an abstract `generate()` method).
5.  **Mock Implementation**: Create `components/mock.py` implementing the base classes with print statements and dummy returns.
6.  **State Management**: Implement `StateManager` to read/write JSON state.
7.  **Orchestrator**: Assemble the pieces. Write the `run_loop()` method that calls the components in sequence.
8.  **CLI**: Use `Typer` in `main.py` to expose commands like `init` (create config) and `run` (start orchestrator).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Validation**: Test that `config.yaml` with missing fields raises Pydantic errors. Test that setting `type: mock` correctly selects the `MockComponentConfig`.
*   **State Manager**: Test saving a state, modifying the file on disk (corrupting it), and ensuring the loader handles it gracefully (or raises an error).
*   **Orchestrator Logic**: Mock the component classes (using `unittest.mock`) and verify that `orchestrator.run_loop()` calls them in the correct order (Generate -> Oracle -> Trainer).

### 5.2. Integration Testing (The Mock Loop)
*   **Goal**: Verify the entire wiring without external physics engines.
*   **Procedure**:
    1.  Create a temporary directory.
    2.  Write a `config.yaml` specifying `type: mock` for all components.
    3.  Run the `Orchestrator`.
    4.  **Assert**:
        *   Log files are created.
        *   `workflow_state.json` updates the iteration counter.
        *   Dummy "potential" files are "created" (or at least the path is tracked).
