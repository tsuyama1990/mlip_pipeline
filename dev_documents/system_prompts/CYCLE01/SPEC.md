# Cycle 01 Specification: Core Framework & CLI

## 1. Summary
This cycle establishes the foundational infrastructure for the PYACEMAKER project. The goal is to create a robust, modular, and type-safe skeleton that will support the subsequent development of specialized modules (Oracle, Trainer, Explorer). We will implement the centralized configuration system, the logging infrastructure, the Command Line Interface (CLI) entry point, and the Abstract Base Classes (Interfaces) that define the contracts for all components. Additionally, we will implement "Mock" versions of these components to verify the data flow within the Orchestrator before integrating real external tools.

## 2. System Architecture

### 2.1. File Structure
The following files will be created or modified in this cycle.

```
mlip-pipeline/
├── pyproject.toml
├── README.md
├── **config.yaml**               # [NEW] Default configuration
├── src/
│   └── mlip_autopipec/
│       ├── **__init__.py**
│       ├── **main.py**           # [NEW] CLI Entry Point using Typer
│       ├── config/
│       │   ├── **__init__.py**
│       │   └── **config_model.py** # [NEW] Pydantic models for GlobalConfig
│       ├── orchestration/
│       │   ├── **__init__.py**
│       │   └── **orchestrator.py** # [NEW] Core logic (Dependency Injection)
│       ├── interfaces/
│       │   ├── **__init__.py**
│       │   ├── **explorer.py**     # [NEW] ABC for Explorer
│       │   ├── **oracle.py**       # [NEW] ABC for Oracle
│       │   ├── **trainer.py**      # [NEW] ABC for Trainer
│       │   └── **validator.py**    # [NEW] ABC for Validator
│       ├── infrastructure/
│       │   ├── **__init__.py**
│       │   └── **mocks.py**        # [NEW] Mock implementations of interfaces
│       └── utils/
│           ├── **__init__.py**
│           └── **logging.py**      # [NEW] Centralized logging setup
└── tests/
    ├── **__init__.py**
    ├── unit/
    │   ├── **test_config.py**      # [NEW] Config validation tests
    │   └── **test_orchestrator.py** # [NEW] Orchestrator logic tests
    └── e2e/
        └── **test_cli.py**         # [NEW] CLI smoke tests
```

## 3. Design Architecture

### 3.1. Configuration Model (`src/mlip_autopipec/config/config_model.py`)
We will use **Pydantic** to define a strict schema for the configuration. This ensures that any user errors in `config.yaml` are caught immediately at startup.

*   **`GlobalConfig`**: The root model.
    *   `work_dir`: Path (Directory to store results).
    *   `max_cycles`: int (Number of active learning iterations).
    *   `random_seed`: int (For reproducibility).
    *   `explorer`: `ExplorerConfig` (Polymorphic config for MD/Random).
    *   `oracle`: `OracleConfig` (DFT settings).
    *   `trainer`: `TrainerConfig` (Pacemaker settings).

### 3.2. Interfaces (`src/mlip_autopipec/interfaces/`)
We use Python's `abc` module to define strict interfaces.

*   **`BaseExplorer`**:
    *   `explore(current_potential, dataset) -> exploration_result`
*   **`BaseOracle`**:
    *   `label(structures) -> labeled_structures`
*   **`BaseTrainer`**:
    *   `train(dataset, validation_set) -> potential_path`
*   **`BaseValidator`**:
    *   `validate(potential_path) -> validation_result`

### 3.3. Orchestrator (`src/mlip_autopipec/orchestration/orchestrator.py`)
The Orchestrator follows the **Dependency Injection** pattern. It does not instantiate concrete classes (like `LammpsDynamics` or `EspressoOracle`) directly. Instead, it receives instances of the Interfaces. This allows us to easily swap in `MockExplorer` or `MockOracle` for testing.

## 4. Implementation Approach

1.  **Setup Project Skeleton**: Create the directory structure as defined in System Architecture.
2.  **Implement Logging**: Create `utils/logging.py` to configure a standard logger (stdout + file).
3.  **Define Configuration**: Write the Pydantic models in `config/config_model.py`.
4.  **Define Interfaces**: Write the ABCs in `interfaces/`. Use `abc.abstractmethod` to enforce implementation.
5.  **Implement Mocks**: Create `infrastructure/mocks.py` where each class implements an Interface but does nothing (or returns dummy data).
    *   `MockExplorer`: Returns a random list of dummy Atoms.
    *   `MockOracle`: Adds random energy/forces to the Atoms.
    *   `MockTrainer`: Creates a dummy `potential.yace` file.
6.  **Implement Orchestrator**: Write the loop logic in `orchestrator.py` using the interfaces.
7.  **Implement CLI**: Use `typer` in `main.py` to:
    *   Load `config.yaml`.
    *   Instantiate the Mocks (for now).
    *   Inject them into the Orchestrator.
    *   Run the loop.
8.  **Create Tests**: Write pytest files.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Validation**: Create `tests/unit/test_config.py`. Feed it valid and invalid dictionaries. Assert that `ValidationError` is raised for missing required fields (e.g., `max_cycles`).
*   **Orchestrator Logic**: Create `tests/unit/test_orchestrator.py`. Instantiate `Orchestrator` with Mocks. Run `run_cycle()`. Assert that `explorer.explore()` and `oracle.label()` were called (using `unittest.mock.MagicMock` wrappers if needed, or just checking side effects in the Mocks).

### 5.2. Integration Testing (CLI)
*   **Smoke Test**: Create `tests/e2e/test_cli.py`. Use `typer.testing.CliRunner`.
    *   Run `mlip-pipeline --help`. Assert exit code 0.
    *   Run `mlip-pipeline run --config valid_config.yaml`. Assert exit code 0 and check logs for "Simulation completed".
