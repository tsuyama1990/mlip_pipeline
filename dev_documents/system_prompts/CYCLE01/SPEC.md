# Cycle 01 Specification: Core Framework & Mocks

## 1. Summary
The goal of Cycle 01 is to establish the foundational infrastructure of PYACEMAKER. This includes the project skeleton, configuration loading, command-line interface (CLI), and the core orchestration logic. Crucially, we will implement "Mock" versions of all major components (Generator, Oracle, Trainer, Dynamics) to verify the data flow and state management of the `Orchestrator` without requiring external heavy dependencies (DFT, LAMMPS) at this stage.

## 2. System Architecture

### 2.1 File Structure
**Bold** files are to be created or modified in this cycle.

```ascii
.
├── pyproject.toml
├── README.md
├── src/
│   └── mlip_autopipec/
│       ├── **__init__.py**
│       ├── **main.py**                 # CLI Entry Point (Typer)
│       ├── **factory.py**              # Component Factory
│       ├── core/
│       │   ├── **__init__.py**
│       │   ├── **orchestrator.py**     # Main Orchestration Loop
│       │   └── **state.py**            # Runtime State Management
│       ├── components/
│       │   ├── **__init__.py**
│       │   ├── generator/
│       │   │   ├── **__init__.py**
│       │   │   └── **mock.py**         # Mock Generator
│       │   ├── oracle/
│       │   │   ├── **__init__.py**
│       │   │   └── **mock.py**         # Mock Oracle
│       │   ├── trainer/
│       │   │   ├── **__init__.py**
│       │   │   └── **mock.py**         # Mock Trainer
│       │   ├── dynamics/
│       │   │   ├── **__init__.py**
│       │   │   └── **mock.py**         # Mock Dynamics
│       │   └── validator/
│       │       ├── **__init__.py**
│       │       └── **mock.py**         # Mock Validator
│       ├── domain_models/
│       │   ├── **__init__.py**
│       │   ├── **config.py**           # GlobalConfig Pydantic Model
│       │   ├── **structure.py**        # Structure Pydantic Model
│       │   └── **potential.py**        # Potential Pydantic Model
│       ├── interfaces/
│       │   ├── **__init__.py**
│       │   ├── **generator.py**        # Base Abstract Class
│       │   ├── **oracle.py**           # Base Abstract Class
│       │   ├── **trainer.py**          # Base Abstract Class
│       │   ├── **dynamics.py**         # Base Abstract Class
│       │   └── **validator.py**        # Base Abstract Class
│       └── utils/
│           ├── **__init__.py**
│           └── **logging.py**          # Centralised Logging
└── tests/
    ├── **conftest.py**
    ├── **test_config.py**
    ├── **test_orchestrator.py**
    └── **test_mocks.py**
```

## 3. Design Architecture

### 3.1 Domain Models (`src/mlip_autopipec/domain_models/`)
We use Pydantic V2 for strict validation.

*   **`config.py`**:
    *   `GlobalConfig`: Root configuration object.
    *   Fields: `workdir` (Path), `max_cycles` (int), `logging_level` (str).
    *   Component Configs: `generator_config`, `oracle_config`, etc. (Dict[str, Any] for now, to be refined in later cycles).
*   **`structure.py`**:
    *   `Structure`: Represents an atomic structure.
    *   Fields: `positions` (np.ndarray), `atomic_numbers` (np.ndarray), `cell` (np.ndarray), `pbc` (List[bool]).
    *   `labels`: Optional[Dict[str, Any]] (energy, forces, stress).
*   **`potential.py`**:
    *   `Potential`: Represents a potential file.
    *   Fields: `filepath` (Path), `version` (str), `metadata` (Dict).

### 3.2 Interfaces (`src/mlip_autopipec/interfaces/`)
Abstract Base Classes (ABC) defining the contract for each component.

*   `BaseGenerator.generate(config) -> List[Structure]`
*   `BaseOracle.compute(structures) -> List[Structure]` (with labels)
*   `BaseTrainer.train(dataset) -> Potential`
*   `BaseDynamics.explore(potential) -> List[Structure]` (High uncertainty structures)
*   `BaseValidator.validate(potential) -> Dict`

### 3.3 Orchestrator (`src/mlip_autopipec/core/orchestrator.py`)
The `Orchestrator` class initializes all components via a `Factory` pattern based on the `GlobalConfig`. It runs the main `run()` loop which iterates through `max_cycles`.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `pyproject.toml` (already done) and directory structure.
2.  **Domain Models**: Implement `config.py`, `structure.py`, `potential.py`. Ensure Pydantic is correctly used.
3.  **Interfaces**: Define ABCs in `interfaces/`.
4.  **Mock Components**: Implement `MockGenerator`, `MockOracle`, etc., that return dummy data (e.g., random structures, random energies) and log their execution.
5.  **Factory**: Implement `create_component(type, config)` in `factory.py` to instantiate Mocks or Real components (though only Mocks for now).
6.  **Orchestrator**: Implement the `run_cycle` logic connecting the mocks.
7.  **CLI**: Use `typer` to create the `mlip-pipeline` command which takes a config file path.
8.  **Logging**: Setup `structlog` or standard `logging` in `utils/logging.py`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Config Loading**: Test `GlobalConfig` validates valid YAML and rejects invalid ones (e.g., missing `workdir`).
*   **Orchestrator Logic**: Test `Orchestrator` initializes components correctly.
*   **Mock Behavior**: Verify Mock components return data in the expected format (Pydantic models).

### 5.2 Integration Testing
*   **Full Loop**: Run the CLI with a dummy `config.yaml`.
    *   *Expectation*: The system runs for `max_cycles`, logs "Generating...", "Calculating...", "Training...", and exits with code 0.
    *   *Verification*: Check log output for the sequence of operations.
