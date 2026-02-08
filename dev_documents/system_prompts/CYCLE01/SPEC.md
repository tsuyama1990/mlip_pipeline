# Cycle 01: Core Framework & Mocks

## 1. Summary

The primary objective of Cycle 01 is to establish the **architectural skeleton** of the PYACEMAKER system. We will implement the core infrastructure that enables the "Zero-Config" philosophy, including the global configuration management, logging system, and the abstract base classes for all future components.

To validate this infrastructure without waiting for complex physics engines (like Quantum Espresso or LAMMPS), we will implement **Mock Components**. These mocks will simulate the behavior of real components—generating random structures, returning fake forces, and producing dummy potential files—allowing us to test the **Orchestrator's** logic and data flow immediately.

By the end of this cycle, we will have a fully functional CLI application that can read a `config.yaml`, initialize the system, and run a complete (albeit physically meaningless) active learning loop from start to finish. This "Walking Skeleton" approach ensures that the integration points are defined and tested early, preventing "integration hell" in later cycles.

## 2. System Architecture

We will create the directory structure and the foundational files. Files in **bold** are to be created or significantly modified in this cycle.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**                   # Entry point (Typer CLI)
├── **factory.py**                # Component Factory
├── components/
│   ├── **__init__.py**
│   ├── dynamics/
│   │   ├── **__init__.py**
│   │   ├── **base.py**           # BaseDynamics interface
│   │   └── **mock.py**           # MockDynamics
│   ├── generator/
│   │   ├── **__init__.py**
│   │   ├── **base.py**           # BaseGenerator interface
│   │   └── **mock.py**           # MockGenerator
│   ├── oracle/
│   │   ├── **__init__.py**
│   │   ├── **base.py**           # BaseOracle interface
│   │   └── **mock.py**           # MockOracle
│   ├── trainer/
│   │   ├── **__init__.py**
│   │   ├── **base.py**           # BaseTrainer interface
│   │   └── **mock.py**           # MockTrainer
│   └── validator/
│       ├── **__init__.py**
│       ├── **base.py**           # BaseValidator interface
│       └── **mock.py**           # MockValidator
├── core/
│   ├── **__init__.py**
│   ├── **orchestrator.py**       # Main loop logic
│   ├── **dataset.py**            # Dataset management (merged .pckl handling)
│   └── **state.py**              # State management (Cycle tracking)
├── domain_models/
│   ├── **__init__.py**
│   ├── **config.py**             # GlobalConfig Pydantic model
│   ├── **structure.py**          # Structure Pydantic model
│   └── **potential.py**          # Potential Pydantic model
├── interfaces/
│   ├── **__init__.py**
│   └── **base_component.py**     # Root interface
└── utils/
    ├── **__init__.py**
    └── **logging.py**            # Logging configuration
```

## 3. Design Architecture

### 3.1. Configuration Management (`domain_models/config.py`)
We use `pydantic` to strictly validate the configuration. The `GlobalConfig` is a singleton-like object passed to all components.

-   **`GlobalConfig`**:
    -   `workdir`: Path (The root of all outputs).
    -   `max_cycles`: int (Stop condition).
    -   `logging_level`: str (INFO, DEBUG).
    -   `components`: Dict (Configuration for each component).

### 3.2. Domain Models
-   **`Structure` (`domain_models/structure.py`)**:
    -   A wrapper around `ase.Atoms` but serializable.
    -   Fields: `positions` (np.ndarray), `atomic_numbers` (np.ndarray), `cell` (np.ndarray), `pbc` (bool/array), `forces` (Optional[np.ndarray]), `energy` (Optional[float]), `stress` (Optional[np.ndarray]).
    -   **Validation**: Ensure arrays match atom counts.
-   **`Potential` (`domain_models/potential.py`)**:
    -   Represents a trained potential.
    -   Fields: `path` (Path), `format` (str="yace"), `metrics` (Dict).

### 3.3. Interfaces (`interfaces/` & `components/*/base.py`)
Each component type has a strict interface.
-   **`BaseGenerator`**: `generate(n_structures: int, config: Dict) -> List[Structure]`
-   **`BaseOracle`**: `compute(structures: List[Structure]) -> List[Structure]` (Returns labeled structures).
-   **`BaseTrainer`**: `train(dataset: Dataset, previous_potential: Optional[Potential]) -> Potential`
-   **`BaseDynamics`**: `explore(potential: Potential, start_structures: List[Structure]) -> List[Structure]` (Returns uncertain structures).

### 3.4. Mock Implementation
-   **`MockGenerator`**: Returns random atoms in a box.
-   **`MockOracle`**: Assigns random forces and energies to the input structures.
-   **`MockTrainer`**: Writes a dummy empty file to `path` and returns a `Potential` object.
-   **`MockDynamics`**: Randomly selects a subset of input structures as "uncertain" and returns them.

## 4. Implementation Approach

1.  **Setup Project**: Initialize `pyproject.toml` (already done), create directory structure.
2.  **Utils**: Implement `utils/logging.py` to setup `structlog` or standard `logging` with a nice format.
3.  **Domain Models**: Implement `structure.py` and `config.py` with Pydantic. Ensure JSON serialization works for Numpy arrays.
4.  **Interfaces**: Define the abstract base classes in `components/*/base.py`.
5.  **Mock Components**: Implement the `Mock*` classes.
6.  **Orchestrator**: Implement the `Orchestrator` class in `core/orchestrator.py`.
    -   `__init__`: Load config, instantiate components via `factory.py`.
    -   `run()`: The `while cycle < max_cycles:` loop.
        -   Step 1: `dynamics.explore()` (or `generator` for cycle 0).
        -   Step 2: `oracle.compute()`.
        -   Step 3: `dataset.append()`.
        -   Step 4: `trainer.train()`.
        -   Step 5: `validator.validate()` (Mock).
7.  **CLI**: Implement `main.py` using `typer`.
    -   `main run config.yaml`: Loads config and starts Orchestrator.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Validation**: Create `tests/test_config.py`. Try loading invalid YAMLs (missing fields, wrong types) and assert `ValidationError` is raised.
-   **Structure Serialization**: Create `tests/test_structure.py`. Create a `Structure`, serialize to dict/JSON, deserialize back, and compare `np.allclose` for positions.
-   **Factory**: Test that `factory.get_component("mock_generator")` returns an instance of `MockGenerator`.

### 5.2. Integration Testing (The "Walking Skeleton")
-   **Scenario**: Run the full loop with Mocks.
-   **Test File**: `tests/test_integration_mock.py`
-   **Procedure**:
    1.  Create a temporary `config.yaml` specifying `mock` implementations for all components.
    2.  Set `max_cycles = 2`.
    3.  Run `Orchestrator(config).run()`.
    4.  **Assert**:
        -   The loop completes without raising exceptions.
        -   `workdir` is created.
        -   `workdir/cycle_01/potential.yace` (dummy file) exists.
        -   `workdir/cycle_02/potential.yace` exists.
