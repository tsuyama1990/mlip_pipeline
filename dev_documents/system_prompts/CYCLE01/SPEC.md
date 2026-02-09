# Cycle 01 Specification: Core Framework

## 1. Summary
This cycle establishes the foundational infrastructure for the PyAceMaker system. The goal is to create the "spinal cord" of the application: the central Orchestrator, the configuration management system, the logging infrastructure, and the abstract interfaces (ABCs) that all future components will implement. By the end of this cycle, the system should be able to read a `config.yaml`, validate it against a strict schema, and instantiate a dummy pipeline that logs its initialization.

## 2. System Architecture

The following file structure will be implemented. Files in **bold** are to be created in this cycle.

```ascii
src/
└── **mlip_autopipec**/
    ├── **__init__.py**
    ├── **main.py**                   # Entry point
    ├── **factory.py**                # Component Factory
    ├── **core**/
    │   ├── **__init__.py**
    │   ├── **orchestrator.py**       # The Brain (Skeletal)
    │   ├── **config_manager.py**     # Config Loader
    │   └── **exceptions.py**         # Custom Exceptions
    ├── **domain_models**/
    │   ├── **__init__.py**
    │   ├── **config.py**             # Pydantic Schemas for Config
    │   ├── **structure.py**          # Pydantic Schemas for Atoms
    │   └── **enums.py**              # Enums (ComponentType, etc.)
    ├── **components**/
    │   ├── **__init__.py**
    │   ├── **base.py**               # Abstract Base Classes
    │   ├── **generator**/            # (Placeholder)
    │   │   └── **__init__.py**
    │   ├── **oracle**/               # (Placeholder)
    │   │   └── **__init__.py**
    │   ├── **trainer**/              # (Placeholder)
    │   │   └── **__init__.py**
    │   ├── **dynamics**/             # (Placeholder)
    │   │   └── **__init__.py**
    │   └── **validator**/            # (Placeholder)
    │       └── **__init__.py**
    └── **utils**/
        ├── **__init__.py**
        ├── **logging.py**            # Centralized Logging
        └── **file_ops.py**           # Safe file operations
```

## 3. Design Architecture

### 3.1 Domain Models (`src/mlip_autopipec/domain_models/`)
We utilize Pydantic V2 for strict type validation.

**`config.py`**:
*   Defines the root `Config` object which mirrors the structure of `config.yaml`.
*   Includes sub-models: `GeneratorConfig`, `OracleConfig`, `TrainerConfig`, `DynamicsConfig`, `ValidatorConfig`.
*   **Constraints:**
    *   Path fields must be validated to ensure they don't point to dangerous locations (e.g., `/`).
    *   Numerical fields (e.g., `cutoff`, `temperature`) must have `ge` (greater or equal) constraints.

**`structure.py`**:
*   Defines the `Structure` object.
*   **Fields:**
    *   `atoms`: An `ase.Atoms` object (Note: Pydantic needs `arbitrary_types_allowed=True` or a custom validator for this).
    *   `provenance`: String tracking where this structure came from (e.g., "MD_HALT_ITER_001").
    *   `features`: Dict storing calculated features (e.g., band gap).

**`enums.py`**:
*   `ComponentRole`: GENERATOR, ORACLE, TRAINER, DYNAMICS, VALIDATOR.
*   `OracleType`: QE, VASP, MOCK.
*   `DynamicsType`: LAMMPS, EON, MOCK.

### 3.2 Abstract Base Classes (`src/mlip_autopipec/components/base.py`)
All components must inherit from `BaseComponent`.

```python
class BaseComponent(ABC):
    @abstractmethod
    def __init__(self, config: BaseModel):
        pass
```

Specific Interfaces:
*   `BaseGenerator`: `generate(n_structures: int) -> List[Structure]`
*   `BaseOracle`: `compute(structures: List[Structure]) -> List[Structure]`
*   `BaseTrainer`: `train(dataset: Path) -> Path`
*   `BaseDynamics`: `explore(potential: Path) -> ExplorationResult`
*   `BaseValidator`: `validate(potential: Path) -> ValidationReport`

### 3.3 Core Logic (`src/mlip_autopipec/core/`)

**`orchestrator.py`**:
*   Class `Orchestrator`.
*   **Attributes:**
    *   `config`: The loaded `Config` object.
    *   `components`: A dictionary mapping `ComponentRole` to instantiated component objects.
*   **Methods:**
    *   `__init__(config_path: str)`: Loads config, initializes logging, and uses `ComponentFactory` to instantiate components.
    *   `run()`: A placeholder method that will eventually run the loop. For now, it just logs "Pipeline initialized".

**`factory.py`**:
*   Implements a simple Factory Pattern to instantiate classes based on the config strings (e.g., `oracle: { type: "QE" }` -> returns `QEOracle` instance).
*   For Cycle 01, we will only implement `Mock` versions of these components in their respective folders or just use the Base classes if appropriate (though Base classes are abstract). Actually, we should create `MockGenerator`, `MockOracle`, etc., in this cycle to prove the wiring works.

## 4. Implementation Approach

1.  **Setup Environment**: Ensure `pyproject.toml` is active.
2.  **Create Domain Models**: Implement `enums.py`, `config.py`, `structure.py`. Verify Pydantic validation works.
3.  **Implement Logging**: Create `utils/logging.py` using `loguru` or standard `logging`. Ensure it writes to both console (INFO) and file (DEBUG).
4.  **Define Interfaces**: Write `components/base.py` with ABCs.
5.  **Create Mock Components**: Create simple implementations (e.g., `MockOracle`) that just log their actions.
6.  **Implement Factory**: Write `factory.py` to dispatch based on config.
7.  **Implement Orchestrator**: Wire everything together.
8.  **Entry Point**: Create `main.py` that accepts a CLI argument (`--config`) and starts the Orchestrator.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Config Validation**: Create valid and invalid `config.yaml` files. Assert `Config.model_validate()` raises `ValidationError` on bad inputs (e.g., negative temperature, missing required fields).
*   **Factory Logic**: Test that `ComponentFactory.create("oracle", {"type": "MOCK"})` returns an instance of `MockOracle`.

### 5.2 Integration Testing
*   **Pipeline Initialization**: Run `python src/mlip_autopipec/main.py --config tests/data/dummy_config.yaml`.
*   **Success Criteria**:
    *   Process exits with code 0.
    *   Log file is created.
    *   Logs contain "Orchestrator initialized successfully" and "Components loaded: [Generator, Oracle, ...]".
