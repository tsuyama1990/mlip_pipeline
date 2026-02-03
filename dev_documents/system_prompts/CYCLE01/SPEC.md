# Cycle 01 Specification: Core Framework & Infrastructure

## 1. Summary

Cycle 01 lays the foundational bedrock for the PYACEMAKER system. Before any scientific logic (DFT, MD, ML) can be implemented, we must establish a robust software engineering infrastructure. This cycle focuses on creating the application skeleton, defining the strict configuration schemas using Pydantic, implementing a centralized logging system, and defining the Abstract Base Classes (Protocols) that will govern the interactions between future components. The goal is to produce a "walking skeleton": an application that can be installed, configured, and run (even if it does nothing but log "Hello World" and instantiate mock objects).

## 2. System Architecture

This section defines the file structure and the components to be implemented in this cycle.

### 2.1 File Structure

**Bold** files are to be created or modified in this cycle.

```text
.
├── **pyproject.toml**              # Updated with dependencies
├── **README.md**                   # Updated with installation instructions
├── **config.yaml**                 # Example configuration file
├── src/
│   └── **mlip_autopipec/**
│       ├── **__init__.py**
│       ├── **main.py**             # CLI Entry Point
│       ├── **config/**
│       │   ├── **__init__.py**
│       │   └── **config_model.py** # Pydantic schemas for Config
│       ├── **domain_models/**
│       │   └── **__init__.py**     # (Empty for now)
│       ├── **interfaces/**
│       │   ├── **__init__.py**
│       │   └── **core_interfaces.py** # Protocol definitions
│       ├── **orchestration/**
│       │   ├── **__init__.py**
│       │   ├── **orchestrator.py** # Main class skeleton
│       │   └── **mocks.py**        # Mock implementations of interfaces
│       ├── **physics/**
│       │   └── **__init__.py**
│       ├── **utils/**
│       │   ├── **__init__.py**
│       │   └── **logging.py**      # Centralized logging
│       └── **__main__.py**         # Support for python -m mlip_autopipec
└── tests/
    ├── **conftest.py**
    ├── **unit/**
    │   ├── **test_config.py**
    │   └── **test_orchestrator.py**
    └── **integration/**
        └── **test_skeleton_loop.py**
```

### 2.2 Component Interaction

In this cycle, the `Orchestrator` will act as a factory. It will:
1.  Read `config.yaml`.
2.  Validate it against `config_model.py`.
3.  Instantiate `MockExplorer`, `MockOracle`, etc., (defined in `mocks.py`) which adhere to the protocols in `core_interfaces.py`.
4.  Log the startup process via `utils/logging.py`.

## 3. Design Architecture

The design philosophy for this cycle is **"Strict Configuration, Loose Coupling"**.

### 3.1 Pydantic Configuration Models (`config_model.py`)
We use Pydantic V2 to enforce type safety at the edge of the system.
-   **`DFTConfig`**: Settings for Quantum Espresso (e.g., `ecutwfc`, `kpoints`).
-   **`TrainingConfig`**: Settings for Pacemaker (e.g., `max_generations`, `cutoff`).
-   **`ExplorationConfig`**: Settings for MD/MC (e.g., `max_temperature`, `steps`).
-   **`SimulationConfig`**: The root model aggregating the above.
-   **Constraints**:
    -   `extra = "forbid"`: Prevent typo-induced silent errors.
    -   `Field(..., description="...")`: All fields must be documented for automatic help generation.

### 3.2 Interfaces (`core_interfaces.py`)
We use `typing.Protocol` to define structural subtyping.
-   **`Explorer`**: Must implement `explore(current_potential) -> List[Structure]`.
-   **`Oracle`**: Must implement `compute(structures) -> List[LabelledStructure]`.
-   **`Trainer`**: Must implement `train(dataset) -> PotentialPath`.
-   **`Validator`**: Must implement `validate(potential) -> ValidationReport`.
-   **Rationale**: By coding against protocols, we decouple the Orchestrator from specific implementations (like QE or LAMMPS), allowing us to swap them for Mocks in tests.

### 3.3 Logging Strategy
-   Using Python's standard `logging` module but configured to output JSON (for machine parsing) or colored text (for human reading).
-   Logs must capture: Timestamp, Module, LogLevel, and Message.
-   Debug logs should trace the state transitions of the Orchestrator.

## 4. Implementation Approach

1.  **Project Initialization**:
    -   Update `pyproject.toml` with `pydantic`, `typer`, `pyyaml`.
    -   Create the directory tree.

2.  **Configuration Implementation**:
    -   Define the Pydantic models in `src/mlip_autopipec/config/config_model.py`.
    -   Write a helper function to load YAML and parse it into `SimulationConfig`.

3.  **Interface Definition**:
    -   Define the Protocols in `src/mlip_autopipec/interfaces/core_interfaces.py`.
    -   Add type hints for `ase.Atoms` where appropriate.

4.  **Mock Implementation**:
    -   Create `src/mlip_autopipec/orchestration/mocks.py`.
    -   Implement `MockExplorer`, `MockOracle`, etc., that return dummy data (e.g., random numbers) but satisfy the interface type checks.

5.  **Orchestrator Logic**:
    -   Implement `Orchestrator.__init__` in `src/mlip_autopipec/orchestration/orchestrator.py`.
    -   It should accept a `SimulationConfig` and initialize the components (using Mocks for now, or based on a factory pattern).

6.  **CLI**:
    -   Implement `main.py` using `typer`.
    -   Command: `pyacemaker run config.yaml`.

## 5. Test Strategy

### 5.1 Unit Testing
-   **`test_config.py`**:
    -   Create valid and invalid `config.yaml` snippets.
    -   Assert that `SimulationConfig.model_validate()` passes for valid inputs.
    -   Assert that it raises `ValidationError` for missing fields or wrong types.
-   **`test_orchestrator.py`**:
    -   Test that `Orchestrator` initializes correctly.
    -   Verify that it logs the correct startup messages.

### 5.2 Integration Testing
-   **`test_skeleton_loop.py`**:
    -   This is the critical "Walking Skeleton" test.
    -   Construct a minimal valid config in code.
    -   Instantiate the `Orchestrator`.
    -   Call a method `run_cycle()` (even if it's empty).
    -   **Success Criteria**: The test runs without raising an exception and produces a log file.
