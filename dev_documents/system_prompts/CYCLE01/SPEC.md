# Cycle 01 Specification: Core Infrastructure & Mocks

## 1. Summary
This cycle establishes the foundational skeleton of the **PYACEMAKER** system. The goal is to create the project structure, define the core domain models and interfaces, and implement a functional CLI that orchestrates "Mock" components. This ensures the high-level logic (the "Orchestrator") is sound before integrating complex external dependencies like Quantum Espresso or LAMMPS.

## 2. System Architecture

### 2.1. File Structure
The following file structure must be created. **Bold** files are to be implemented in this cycle.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**                 # CLI Entry point (Typer)
├── **factory.py**              # Component Factory
├── **constants.py**            # Global constants
├── domain_models/
│   ├── **__init__.py**
│   ├── **structure.py**        # Structure Pydantic Model
│   ├── **potential.py**        # Potential Pydantic Model
│   └── **config.py**           # GlobalConfig Pydantic Model
├── interfaces/
│   ├── **__init__.py**
│   ├── **generator.py**        # Abstract Base Class
│   ├── **oracle.py**           # Abstract Base Class
│   ├── **trainer.py**          # Abstract Base Class
│   └── **dynamics.py**         # Abstract Base Class
├── core/
│   ├── **__init__.py**
│   ├── **orchestrator.py**     # Main Logic (using Mocks)
│   └── **state.py**            # State Management
└── infrastructure/
    ├── **__init__.py**
    └── **mocks.py**            # Mock Implementations of Interfaces
tests/
    ├── **conftest.py**
    ├── **test_cli.py**
    ├── **test_orchestrator.py**
    └── **test_domain_models.py**
```

### 2.2. Class Blueprints

#### `src/mlip_autopipec/interfaces/generator.py`
```python
from abc import ABC, abstractmethod
from typing import Iterator
from mlip_autopipec.domain_models.structure import Structure

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, count: int) -> Iterator[Structure]:
        """Generate candidate structures."""
        pass
```

#### `src/mlip_autopipec/interfaces/oracle.py`
```python
from abc import ABC, abstractmethod
from typing import Iterable, Iterator
from mlip_autopipec.domain_models.structure import Structure

class BaseOracle(ABC):
    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """Compute properties (energy, forces) for structures."""
        pass
```

## 3. Design Architecture

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/`)
We use Pydantic V2 for strict validation.

*   **`Structure`**: Represents an atomic configuration.
    *   Fields: `atomic_numbers` (List[int]), `positions` (List[List[float]]), `cell` (List[List[float]]), `pbc` (List[bool]), `energy` (Optional[float]), `forces` (Optional[List[List[float]]]).
    *   Validation: Ensure array shapes match (N_atoms).
*   **`GlobalConfig`**: The root configuration object.
    *   Fields: `workdir` (Path), `max_cycles` (int), `generator` (dict), `oracle` (dict), `trainer` (dict), `dynamics` (dict).
    *   Validation: Verify paths exist or are creatable.

### 3.2. Mock Components (`src/mlip_autopipec/infrastructure/mocks.py`)
Implement `MockGenerator`, `MockOracle`, `MockTrainer`, `MockDynamics` inheriting from their respective ABCs.
*   **`MockOracle`**: Instead of running DFT, it assigns random energy/forces to the input structure and returns it.
*   **`MockTrainer`**: Creates a dummy file named `potential_cycle_X.yace` without running Pacemaker.
*   **`MockDynamics`**: Returns a list of "high uncertainty" dummy structures to simulate Active Learning feedback.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `pyproject.toml` (already done) and directory structure.
2.  **Domain Models**: Implement `structure.py` and `config.py` with Pydantic.
3.  **Interfaces**: Define ABCs in `interfaces/`.
4.  **Mocks**: Implement the Mock classes in `infrastructure/mocks.py`.
5.  **Orchestrator**: Implement `Orchestrator.run()` in `core/orchestrator.py`. It should instantiate components based on config (using `factory.py`) and run the loop:
    *   Generator -> Oracle -> Trainer -> Dynamics -> (Loop).
6.  **CLI**: Implement `main.py` using `typer`. It should read a YAML config, instantiate `Orchestrator`, and call `run()`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_domain_models.py`**: Verify Pydantic validation (e.g., passing wrong array shapes raises ValidationError).
*   **`test_cli.py`**: Verify `mlip-pipeline run config.yaml` executes without error code.

### 5.2. Integration Testing (Mocked)
*   **`test_orchestrator.py`**:
    *   Instantiate `Orchestrator` with a `MockConfig`.
    *   Run `orchestrator.run_cycle()`.
    *   Assert that `MockOracle.compute()` was called.
    *   Assert that `MockTrainer.train()` created a dummy potential file.
