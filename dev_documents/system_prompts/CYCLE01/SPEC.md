# Cycle 01: Core Framework & Orchestrator Skeleton

## 1. Summary

Cycle 01 lays the foundation for the PYACEMAKER project. The primary objective is to establish the project structure, implement the central control logic (Orchestrator), and define the configuration schemas using Pydantic. We will also implement "Mock" versions of all major components (Generator, Oracle, Trainer, Dynamics).

This cycle allows us to verify the "Control Plane" of the architecture. We will prove that the system can read a configuration file, instantiate the correct components, and execute a simulated active learning loop without needing to install heavy external dependencies like LAMMPS or Quantum Espresso. This "Mock Mode" is crucial for CI/CD and rapid development.

## 2. System Architecture

The following file structure will be created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── **pyproject.toml**            # Project dependencies and tool config
├── **README.md**                 # Project documentation
├── **config.yaml**               # Example User configuration
├── src/
│   └── mlip_autopipec/
│       ├── **__init__.py**
│       ├── **main.py**           # CLI Entry point
│       ├── core/
│       │   ├── **__init__.py**
│       │   ├── **orchestrator.py** # Main event loop
│       │   ├── **config.py**     # Pydantic models
│       │   └── **logger.py**     # Logging configuration
│       ├── domain_models/
│       │   ├── **__init__.py**
│       │   ├── **inputs.py**     # CandidateStructure
│       │   ├── **results.py**    # ValidationMetrics
│       │   └── **enums.py**      # Component Types
│       ├── components/
│       │   ├── **__init__.py**
│       │   ├── **base.py**       # Abstract Base Classes
│       │   └── **mock.py**       # Mock Implementations
│       ├── utils/
│       │   ├── **__init__.py**
│       │   └── **file_manager.py**
│       └── **constants.py**
└── tests/
    ├── **conftest.py**
    ├── **test_config.py**
    └── **test_orchestrator.py**
```

## 3. Design Architecture

### Configuration Management (`core/config.py`)
We use `pydantic` to define strict schemas for the configuration. This ensures type safety and provides helpful error messages to the user.

*   `GeneratorConfig`: Defines parameters for structure generation (e.g., `max_atoms`, `composition`).
*   `OracleConfig`: Defines DFT parameters (e.g., `calculator_type`, `kpoints`).
*   `TrainerConfig`: Defines ACE training parameters.
*   `OrchestratorConfig`: The root configuration object that nests the above configs.

### Component Interfaces (`components/base.py`)
We define Abstract Base Classes (ABCs) to enforce a consistent API.

*   `BaseGenerator`: Must implement `generate(n_structures) -> List[Structure]`.
*   `BaseOracle`: Must implement `compute(structures) -> List[Structure]`.
*   `BaseTrainer`: Must implement `train(dataset) -> PotentialPath`.
*   `BaseDynamics`: Must implement `explore(potential) -> ExplorationResult`.

### Mock Implementations (`components/mock.py`)
These classes implement the ABCs but perform no heavy computation.
*   `MockGenerator`: Returns random atoms using `ase.build.bulk`.
*   `MockOracle`: Assigns random energies and forces to the structures.
*   `MockTrainer`: Creates a dummy `.yace` file.
*   `MockDynamics`: Simulates a run by returning a "Halt" signal with a random probability.

## 4. Implementation Approach

1.  **Project Setup**: Initialize the directory structure and `pyproject.toml` with dependencies (`ase`, `pydantic`, `pyyaml`, `numpy`).
2.  **Domain Modeling**: Implement `domain_models/enums.py` and `inputs.py` to define the data structures passed between components.
3.  **Base Classes**: Define the ABCs in `components/base.py`.
4.  **Configuration**: Implement `core/config.py` using Pydantic. Ensure it can load from `config.yaml`.
5.  **Mocks**: Implement the mock components in `components/mock.py`.
6.  **Orchestrator**: Implement `core/orchestrator.py`. It should:
    *   Load the config.
    *   Instantiate components (switching between Real/Mock based on config).
    *   Run a simple loop: Generate -> Oracle -> Train -> Dynamics.
7.  **CLI**: Implement `main.py` using `argparse` to trigger the Orchestrator.

## 5. Test Strategy

### Unit Testing
*   **Config Test**: Create a valid `config.yaml` and assert that `OrchestratorConfig` loads it without error. Create an invalid config (missing fields) and assert that it raises `ValidationError`.
*   **Mock Component Test**: Instantiate `MockOracle` and call `compute()`. Assert that the returned structures have `energy` and `forces` arrays of the correct shape.

### Integration Testing
*   **Orchestrator Loop**: Run the Orchestrator in "Mock Mode" for 2 cycles.
    *   Verify that "dummy" potential files are created in the output directory.
    *   Verify that the log file contains expected messages ("Starting cycle 1", "Mock training complete").
    *   Ensure the process exits with code 0.
