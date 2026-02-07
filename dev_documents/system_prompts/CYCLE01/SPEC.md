# Cycle 01 Specification: Foundation & Orchestrator

## 1. Summary

Cycle 01 lays the cornerstone of the PYACEMAKER project. The primary objective is to establish a robust, modular software architecture that can support the complex "Active Learning" workflows envisioned in the final system. Instead of attempting to implement the heavy physics engines (DFT, MD) immediately, this cycle focuses on the **Orchestrator**—the central nervous system—and defines the strict **Interfaces** (contracts) that all future components must adhere to.

By the end of this cycle, we will have a fully functional CLI application that can parse a user's configuration, initialize the system components, and execute the complete learning loop (Exploration -> Labelling -> Training -> Validation) using **Mock Implementations**. These mocks will simulate the time delays and data structures of real physics codes without requiring any external dependencies (like Quantum Espresso or LAMMPS). This allows us to verify the logic, error handling, logging, and data flow of the Orchestrator before introducing the complexity of quantum mechanics.

This "Skeleton First" approach ensures that the system is architecturally sound, testable, and ready for the parallel development of specific modules in subsequent cycles. It also establishes the project's development culture: strict type hinting (Mypy), automated formatting (Ruff), and comprehensive testing (Pytest).

## 2. System Architecture

This section defines the file structure and the code blueprints for the foundation.

### 2.1. File Structure (ASCII Tree)

Files to be created in this cycle are marked in **bold**.

```
PYACEMAKER/
├── **pyproject.toml**              # Dependency management
├── **README.md**                   # Project documentation
├── **src/**
│   └── **mlip_autopipec/**
│       ├── **__init__.py**
│       ├── **main.py**             # CLI Entry Point (Typer)
│       ├── **config/**             # Configuration Definitions
│       │   ├── **__init__.py**
│       │   └── **base_config.py**  # Pydantic Models
│       ├── **domain_models/**      # Data Transfer Objects
│       │   ├── **__init__.py**
│       │   ├── **structure.py**    # Mock Structure definition
│       │   └── **potential.py**    # Mock Potential definition
│       ├── **interfaces/**         # Abstract Base Classes
│       │   ├── **__init__.py**
│       │   ├── **abstract_orchestrator.py**
│       │   ├── **abstract_structure_generator.py**
│       │   ├── **abstract_oracle.py**
│       │   ├── **abstract_trainer.py**
│       │   ├── **abstract_dynamics.py**
│       │   └── **abstract_validator.py**
│       ├── **orchestrator/**       # Core Logic
│       │   ├── **__init__.py**
│       │   └── **simple_orchestrator.py**
│       ├── **infrastructure/**     # Concrete Implementations
│       │   ├── **__init__.py**
│       │   └── **mocks.py**        # Mock implementations of all interfaces
│       └── **utils/**
│           ├── **__init__.py**
│           └── **logging.py**      # Centralized logging config
└── **tests/**
    ├── **__init__.py**
    ├── **conftest.py**
    ├── **unit/**
    │   ├── **test_config.py**
    │   └── **test_orchestrator.py**
    └── **integration/**
        └── **test_full_cycle_mock.py**
```

### 2.2. Component Blueprints

**1. `src/mlip_autopipec/main.py`**:
The entry point using `typer`.
```python
import typer
from mlip_autopipec.orchestrator.simple_orchestrator import SimpleOrchestrator
from mlip_autopipec.config.base_config import GlobalConfig

app = typer.Typer()

@app.command()
def run(config_path: str = "config.yaml"):
    """
    Starts the Active Learning Orchestrator with the given config.
    """
    # Load Config
    config = GlobalConfig.from_yaml(config_path)
    # Init Orchestrator
    orchestrator = SimpleOrchestrator(config)
    # Run
    orchestrator.run()

if __name__ == "__main__":
    app()
```

**2. `src/mlip_autopipec/interfaces/abstract_oracle.py`**:
Example of a strict interface using `ABC`.
```python
from abc import ABC, abstractmethod
from typing import List
from mlip_autopipec.domain_models.structure import Structure

class BaseOracle(ABC):
    @abstractmethod
    def compute(self, structures: List[Structure]) -> List[Structure]:
        """
        Takes a list of structures and returns them with computed
        energy, forces, and stress.
        """
        pass
```

## 3. Design Architecture

### 3.1. Domain Models (Pydantic)
We use Pydantic to strictly define the data passing through the system.

*   **`Structure`**: In this cycle, it can be a simple wrapper around a dictionary or a placeholder class. It must have fields for `positions`, `cell`, `energy` (Optional), and `forces` (Optional).
*   **`GlobalConfig`**: The root configuration. It must be hierarchical.
    ```python
    class OracleConfig(BaseModel):
        type: Literal["mock", "quantum_espresso"] = "mock"
        # ... other dft params

    class GlobalConfig(BaseModel):
        project_name: str
        oracle: OracleConfig
        # ... other components
    ```

### 3.2. The Orchestrator Logic
The `SimpleOrchestrator` implements the State Machine.
*   **State 1: Initialization**: Instantiates `MockOracle`, `MockTrainer`, etc., based on the config.
*   **State 2: Loop**:
    *   Call `structure_generator.get_candidates()`
    *   Call `oracle.compute()`
    *   Call `trainer.train()`
    *   Call `dynamics.run_exploration()` -> returns `ExplorationResult` (halted or not).
    *   If `halted`, add new structure to dataset and repeat.
*   **State 3: Termination**: If max iterations reached or validation passes.

### 3.3. Mocking Strategy
The `src/mlip_autopipec/infrastructure/mocks.py` file allows us to test the *orchestration* without the *physics*.
*   **`MockOracle`**: Sleeps for 0.1s and assigns random energy values to the structures.
*   **`MockTrainer`**: Writes a dummy `.yace` file to the disk.
*   **`MockDynamics`**: Returns a random "Halt" status to simulate uncertainty detection.

## 4. Implementation Approach

1.  **Project Setup**: Initialize git, create the directory structure, and setup `pyproject.toml` with `ruff`, `mypy`, `pytest`.
2.  **Domain & Interfaces**: Define the `Structure` data class and the `ABC`s for all components. This locks down the API.
3.  **Config**: Implement the Pydantic models to parse a dummy `config.yaml`.
4.  **Mocks**: Implement the "Happy Path" mocks in `mocks.py`.
5.  **Orchestrator**: Write the loop logic in `simple_orchestrator.py`, wiring the components together.
6.  **CLI**: Expose the orchestrator via `typer` in `main.py`.
7.  **Testing**: Write the integration test that runs the whole loop with mocks.

## 5. Test Strategy

### 5.1. Unit Testing Approach (Min 300 words)
Unit tests in Cycle 01 focus on the correctness of the configuration parsing and the robustness of the Orchestrator's state machine logic.
*   **Config Tests**: We will create various invalid YAML files (missing fields, wrong types) and verify that `GlobalConfig` raises informative `ValidationError`s. This ensures the system fails early and helpfully if the user makes a typo.
*   **Factory Tests**: We will test the logic that instantiates classes based on strings (e.g., `type: "mock"` -> `MockOracle`). We must ensure that adding a new type in the future is easy and that unknown types raise clear errors.
*   **Logging Tests**: We will verify that the logging utility correctly routes messages to both the console and a file, and that the verbosity levels work as expected.

### 5.2. Integration Testing Approach (Min 300 words)
The integration test `test_full_cycle_mock.py` is the most critical deliverable of this cycle.
*   **The "Dry Run"**: This test will programmatically create a `GlobalConfig` object set to use "mock" components for everything. It will then instantiate the `SimpleOrchestrator` and call `.run()`.
*   **Assertions**:
    *   **Completion**: The `.run()` method returns successfully (exit code 0).
    *   **File Creation**: The test checks that the expected directory hierarchy (`active_learning/iter_001/`) was created.
    *   **Data Flow**: We will inspect the "dummy" output files (e.g., the mock potential file) to ensure they were "generated" by the MockTrainer.
    *   **Loop Logic**: By configuring the `MockDynamics` to return "Halt" for the first 2 calls and "Converged" for the 3rd, we verify that the Orchestrator correctly loops 3 times and then proceeds to the next stage.
