# Cycle 01 Specification: Foundation & Mocks

## 1. Summary

**Goal**: Establish the foundational architecture of the PYACEMAKER system. This cycle focuses on defining the core data structures (Domain Models), abstract interfaces for all major components, and providing "Mock" implementations. These mocks allow the development of the Orchestrator logic in Cycle 02 without depending on external physics codes (Quantum Espresso, Pacemaker, LAMMPS) which are heavy and complex to set up in a CI environment.

**Key Deliverables**:
1.  **Domain Models**: Pydantic definitions for `Structure`, `Potential`, `GlobalConfig`.
2.  **Interfaces**: Abstract Base Classes (ABCs) for `Oracle`, `Trainer`, `Dynamics`, `StructureGenerator`, `Validator`, `Selector`.
3.  **Mock Implementations**: Functional mocks for all interfaces that simulate their behaviour (e.g., `MockOracle` returns random energies, `MockDynamics` returns a "halt" status after N steps).

## 2. System Architecture

The following file structure will be created. Files in **bold** are the primary focus of this cycle.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**               # Basic CLI skeleton
├── domain_models/
│   ├── **__init__.py**
│   ├── **config.py**         # GlobalConfig, OracleConfig, etc.
│   ├── **structure.py**      # Structure, AtomsData
│   └── **potential.py**      # Potential, ExplorationResult
├── interfaces/
│   ├── **__init__.py**
│   ├── **oracle.py**         # BaseOracle
│   ├── **trainer.py**        # BaseTrainer
│   ├── **dynamics.py**       # BaseDynamics
│   ├── **generator.py**      # BaseStructureGenerator
│   ├── **validator.py**      # BaseValidator
│   └── **selector.py**       # BaseSelector
├── infrastructure/
│   ├── **__init__.py**
│   └── **mocks.py**          # MockOracle, MockTrainer, etc.
└── utils/
    ├── **__init__.py**
    └── **logging.py**        # Logging configuration
```

## 3. Design Architecture

### 3.1 Domain Models (Pydantic)

*   **`Structure`**: Represents an atomic configuration.
    *   Fields: `positions` (N,3 float array), `cell` (3,3 float array), `symbols` (List[str]), `pbc` (bool array), `properties` (Dict).
    *   Validation: Ensure array shapes match number of atoms.
    *   Serialisation: Must handle numpy array serialisation to JSON.
*   **`Potential`**: Represents a trained model.
    *   Fields: `path` (Path), `version` (str), `metrics` (Dict).
*   **`GlobalConfig`**: The root configuration object.
    *   Fields: `workdir` (Path), `max_cycles` (int), `oracle` (OracleConfig), `trainer` (TrainerConfig), etc.
    *   Constraint: Uses `Literal` types to select implementation (e.g., `type: Literal["mock", "qe"]`).

### 3.2 Interfaces (ABCs)

All components must inherit from these ABCs to ensure interchangeable implementations.

*   `BaseOracle`: `compute(structures: List[Structure]) -> List[Structure]`
*   `BaseTrainer`: `train(dataset: List[Structure], ...) -> Potential`
*   `BaseDynamics`: `run(potential: Potential, ...) -> ExplorationResult`
*   `BaseStructureGenerator`: `generate(source: Structure, ...) -> List[Structure]`
*   `BaseValidator`: `validate(potential: Potential) -> ValidationResult`

### 3.3 Mock Implementations

*   `MockOracle`: Adds random "energy" and "forces" to the input structures.
*   `MockTrainer`: Creates a dummy file (`dummy.yace`) and returns a `Potential` object pointing to it.
*   `MockDynamics`:
    *   Simulates a run by returning `ExplorationResult`.
    *   Randomly decides to "halt" (high uncertainty) or "converge" based on a configured probability.
    *   Returns a random structure as the "halt structure".

## 4. Implementation Approach

1.  **Setup Project**: Initialise `src/mlip_autopipec` and `tests`.
2.  **Define Models**: Implement `domain_models/` modules using Pydantic V2.
3.  **Define Interfaces**: Create `interfaces/` with `abc.ABC`. Ensure strict type hinting.
4.  **Implement Mocks**: Create `infrastructure/mocks.py` implementing all interfaces.
5.  **Setup Logging**: Implement `utils/logging.py` using standard Python logging.
6.  **CLI Skeleton**: Create `main.py` using `typer` to parse config and load the appropriate components (factory pattern).

## 5. Test Strategy

### 5.1 Unit Testing
*   **Config Loading**: Test loading a valid YAML into `GlobalConfig`. Verify defaults and type validation.
*   **Structure Validation**: Test creating `Structure` with invalid arrays (wrong shape) to ensure Pydantic raises errors.
*   **Mock Behaviour**: Verify `MockOracle` adds properties. Verify `MockTrainer` creates the dummy file.

### 5.2 Integration Testing
*   **Factory Test**: Create a test that instantiates components based on `type="mock"` config.
*   **Mock Loop**: Write a script that manually calls `dynamics -> generator -> oracle -> trainer` using the mock objects to verify data flow compatibility.
