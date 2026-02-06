# Cycle 01 Specification: Foundation & Orchestrator Skeleton

## 1. Summary

The goal of Cycle 01 is to establish the **Core Framework** of the PYACEMAKER system. We will not perform any actual physics simulations or machine learning training in this cycle. Instead, we will build the "Skeleton" of the application: the directory structure, the configuration parser, the logging system, and the central **Orchestrator**.

We will define the **Abstract Base Classes (Interfaces)** for all major components (Explorer, Oracle, Trainer) and provide **Mock Implementations**. This allows us to verify the control flow and data passing logic of the Orchestrator without needing external dependencies like LAMMPS or Quantum Espresso installed.

## 2. System Architecture

We will create the initial file structure. **Bold** files are to be created in this cycle.

```ascii
src/
└── **mlip_autopipec**/
    ├── **__init__.py**
    ├── **main.py**                     # CLI Entry Point using Typer/Click
    ├── **constants.py**                # Global Constants (versions, default paths)
    ├── **config**/
    │   ├── **__init__.py**
    │   └── **config_model.py**         # Pydantic Schemas for GlobalConfig
    ├── **domain_models**/
    │   ├── **__init__.py**
    │   └── **structure.py**            # Domain Models: Dataset, Structure
    ├── **interfaces**/
    │   ├── **__init__.py**
    │   ├── **explorer.py**             # ABC for Explorer
    │   ├── **oracle.py**               # ABC for Oracle
    │   ├── **trainer.py**              # ABC for Trainer
    │   └── **validator.py**            # ABC for Validator
    ├── **infrastructure**/
    │   ├── **__init__.py**
    │   └── **mocks.py**                # Mock implementations for all interfaces
    └── **utils**/
        ├── **__init__.py**
        └── **logging.py**              # Centralized logging setup
tests/
    ├── **test_config.py**
    └── **test_orchestrator.py**
```

## 3. Design Architecture

### 3.1. Configuration Model (Pydantic)
We require a robust configuration system.
**File:** `src/mlip_autopipec/config/config_model.py`

*   `GlobalConfig`: Root model.
    *   `work_dir`: Path (required)
    *   `max_cycles`: int (default 10)
    *   `oracle`: OracleConfig
    *   `trainer`: TrainerConfig
    *   `explorer`: ExplorerConfig
*   `OracleConfig`:
    *   `type`: Literal["mock", "espresso"] (default "mock")
*   `TrainerConfig`:
    *   `type`: Literal["mock", "pacemaker"] (default "mock")

### 3.2. Interfaces (ABCs)
We use `abc.ABC` to enforce contracts.

*   `BaseOracle`:
    *   `compute(structures: List[Atoms]) -> List[Atoms]`
*   `BaseTrainer`:
    *   `train(dataset: Dataset, previous_potential: Optional[Path]) -> Path`
*   `BaseExplorer`:
    *   `explore(potential: Path) -> ExplorationResult`

## 4. Implementation Approach

1.  **Project Setup**: Initialize the package structure.
2.  **Domain Models**: Implement `structure.py` to define what a "Dataset" is (wrapper around list of ASE Atoms).
3.  **Interfaces**: Define the ABCs in `interfaces/`.
4.  **Mocks**: Implement `MockOracle`, `MockTrainer`, `MockExplorer` in `infrastructure/mocks.py`.
    *   `MockOracle`: Adds random numbers to `atoms.info['energy']`.
    *   `MockTrainer`: Creates a dummy file named `potential.yace`.
    *   `MockExplorer`: Returns a fixed "Halt" status after N calls or random decision.
5.  **Configuration**: Implement Pydantic models in `config_model.py`.
6.  **Orchestrator**: Implement the main loop in `main.py` (or a dedicated `Orchestrator` class).
    *   Load Config.
    *   Instantiate components based on config `type`.
    *   Run loop: Explore -> (If Halt) -> Oracle -> Train -> Loop.
7.  **CLI**: Use `argparse` or `typer` in `main.py` to accept a config file path.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Test**: Create a valid YAML and an invalid YAML. Assert `GlobalConfig` parses the valid one and raises `ValidationError` for the invalid one.
*   **Mock Test**: Instantiate `MockOracle`. Pass it some dummy Atoms. Assert it returns Atoms with energy values.

### 5.2. Integration Testing
*   **The "Dry Run"**: Create a test that runs the full `main` function with a config set to "mock" mode.
    *   Assert that the "training" loop runs for the specified number of cycles.
    *   Assert that dummy "potential.yace" files are created in the `work_dir`.
    *   Assert that logs are written.
