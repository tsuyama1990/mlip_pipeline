# CYCLE 01 Specification: Core Framework & Mock Loop

## 1. Summary

This cycle establishes the foundation of the PYACEMAKER system. The primary objective is to implement the **Orchestrator** and the **Skeleton Interface** of the core components (Structure Generator, Oracle, Trainer, Dynamics Engine).

We will verify the architecture by running a "Mock Loop" where:
1.  The `Orchestrator` loads a configuration file.
2.  A `MockExplorer` generates dummy atomic structures.
3.  A `MockOracle` calculates fake energy and forces.
4.  A `MockTrainer` simulates the training process.
5.  The loop repeats for a specified number of cycles.

No real scientific calculations (DFT/MD) will be performed in this cycle. This ensures the control logic and data flow are robust before integrating heavy computational engines.

## 2. System Architecture

The following file structure will be created. **Bold** files are to be implemented in this cycle.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**                     # CLI Entry Point (Typer)
├── config/
│   ├── **__init__.py**
│   └── **config_model.py**         # Pydantic Schemas (GlobalConfig)
├── domain_models/
│   ├── **__init__.py**
│   ├── **structures.py**           # StructureMetadata, Atoms wrapper
│   └── **dataset.py**              # Dataset abstraction
├── interfaces/
│   ├── **__init__.py**
│   └── **core_interfaces.py**      # Protocols (Explorer, Oracle, Trainer)
├── orchestration/
│   ├── **__init__.py**
│   ├── **orchestrator.py**         # Main Loop Logic
│   └── **mocks.py**                # Mock implementations
└── utils/
    ├── **__init__.py**
    └── **logging.py**              # Structured logging setup
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/`)
*   **`StructureMetadata`**: A Pydantic model wrapping `ase.Atoms`. It must strictly validate input types.
    *   Fields: `structure` (Any - validated as ase.Atoms), `source` (str), `generation_method` (str).
*   **`Dataset`**: An abstraction for a collection of labeled structures.

### 3.2. Interfaces (`interfaces/core_interfaces.py`)
We define Python `Protocols` to ensure loose coupling.
*   **`Explorer`**: `generate(config) -> list[StructureMetadata]`
*   **`Oracle`**: `calculate(structures) -> list[StructureMetadata]`
*   **`Trainer`**: `train(dataset, previous_potential) -> Path`
*   **`Validator`**: `validate(potential_path) -> ValidationResult`

### 3.3. Configuration (`config/config_model.py`)
*   **`GlobalConfig`**:
    *   `work_dir`: Path
    *   `max_cycles`: int
    *   `random_seed`: int

### 3.4. Orchestrator (`orchestration/orchestrator.py`)
*   The `Orchestrator` class is initialized with a `GlobalConfig`.
*   It instantiates the components (Mocks for now).
*   It runs a `run_loop()` method that iterates `max_cycles` times.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `pyproject.toml` (already done) and directory structure.
2.  **Domain & Config**: Implement `structure.py` and `config_model.py` using Pydantic. Ensure `ase` is installed.
3.  **Interfaces**: Define the Protocols in `core_interfaces.py`.
4.  **Mocks**: Implement `MockExplorer`, `MockOracle`, etc., in `mocks.py`. They should simply log their actions and return dummy objects.
5.  **Orchestrator**: Implement the logic to glue these mocks together.
6.  **CLI**: Use `typer` to create the `mlip-pipeline` command.
    *   Command: `run --config config.yaml`

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_config.py`**: Verify `GlobalConfig` raises errors on invalid input (e.g., negative `max_cycles`).
*   **`test_domain.py`**: Verify `StructureMetadata` rejects non-Atoms objects.

### 5.2. Integration Testing
*   **`test_skeleton_loop.py`**:
    *   Instantiate `Orchestrator` with Mocks.
    *   Run `orchestrator.run_loop()`.
    *   Assert that the "MockTrainer updated potential" log message appears `max_cycles` times.
