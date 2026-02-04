# Cycle 01 Specification: Core Skeleton & Mock Orchestration

## 1. Summary

The objective of Cycle 01 is to establish the foundational software architecture of **PYACEMAKER**. We will build the "Skeleton" of the system: the directory structure, the configuration management system, the core interfaces (Protocols), and the central `Orchestrator` logic.

To verify the logic without waiting for heavy physical computations (DFT/MD), we will implement **Mock Components** (`MockExplorer`, `MockOracle`, `MockTrainer`). This allows us to run a complete "Dry Run" of the Active Learning Cycle—generating random atoms, simulating force calculations, and pretending to train a model—within seconds. This confirms that the data flow and state transitions are correctly implemented before we introduce complex physics engines.

## 2. System Architecture

We will create the initial project structure. **Bold** files are to be created or significantly modified in this cycle.

```ascii
src/mlip_autopipec/
├── __init__.py
├── main.py                  # Entry point (CLI)
├── config/
│   ├── __init__.py
│   ├── **config_model.py**  # Pydantic Schemas
│   └── **defaults.py**      # Default values
├── domain_models/
│   ├── __init__.py
│   ├── **structures.py**    # StructureMetadata
│   └── **potential.py**     # Potential objects
├── interfaces/
│   ├── __init__.py
│   └── **core_interfaces.py** # Protocols (Explorer, Oracle, etc.)
├── orchestration/
│   ├── __init__.py
│   ├── **orchestrator.py**  # Main Logic
│   └── **mocks.py**         # Mock implementations
├── utils/
│   ├── __init__.py
│   └── **logging.py**       # Logging setup
└── tests/
    ├── __init__.py
    └── integration/
        └── **test_skeleton_loop.py**
```

## 3. Design Architecture

### 3.1. Configuration Schema (`config/config_model.py`)
We use Pydantic to strictly define the system configuration.
*   **`DFTConfig`**: settings for the Oracle (e.g., code, k-points).
*   **`TrainingConfig`**: settings for Pacemaker (e.g., cutoff, potential type).
*   **`ExplorationConfig`**: settings for the structure generator.
*   **`GlobalConfig`**: The root object containing all subsections.

### 3.2. Core Protocols (`interfaces/core_interfaces.py`)
We define Python `Protocols` (Interfaces) to decouple the Orchestrator from concrete implementations.
*   **`Explorer`**: `generate_candidates(policy) -> List[Structure]`
*   **`Oracle`**: `compute(structures) -> List[Structure]` (Adds forces/energy)
*   **`Trainer`**: `train(dataset, initial_potential) -> Potential`
*   **`Validator`**: `validate(potential) -> ValidationResult`

### 3.3. Mock Implementations (`orchestration/mocks.py`)
*   **`MockExplorer`**: Generates a valid `ase.Atoms` object with random positions (using `ase.build.bulk`).
*   **`MockOracle`**: Assigns forces using a simple Lennard-Jones calculator (`ase.calculators.lj`). This ensures the "data" has valid numbers.
*   **`MockTrainer`**: Creates a dummy file named `potential_cycle_XX.yace` and sleeps for 0.1s.

## 4. Implementation Approach

1.  **Project Setup**: Initialize the `src/mlip_autopipec` package structure.
2.  **Domain Modeling**: Implement `StructureMetadata` in `domain_models`. It should wrap `ase.Atoms` to add lineage tracking (where did this structure come from?).
3.  **Interface Definition**: Define the Protocols in `interfaces`. Use `typing.Protocol` and `abc.abstractmethod`.
4.  **Mocking**: Implement the Mock classes. They must adhere strictly to the Protocols.
5.  **Orchestrator Logic**: Implement the `run_cycle()` loop in `orchestrator.py`.
    *   Load Config.
    *   Initialize Components (Mocks for now).
    *   Loop `N` times: Explore -> Compute -> Train -> Validate (Mock).
    *   Use `logging` to trace execution.
6.  **CLI**: Create a simple `main.py` using `typer` or `argparse` to launch the Orchestrator.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Test**: Verify that a valid `config.yaml` is parsed correctly into Pydantic models. Verify that missing fields raise `ValidationError`.
*   **Mock Test**: Verify `MockOracle` adds an array of forces to the atoms object.

### 5.2. Integration Testing (`tests/integration/test_skeleton_loop.py`)
*   **The Skeleton Run**: Instantiates the `Orchestrator` with a test config and runs 3 cycles.
*   **Assertions**:
    *   Check that the loop completed without error.
    *   Check that "potential" files were created in the temporary directory.
    *   Check logs for the sequence: "Exploration finished" -> "DFT finished" -> "Training finished".
