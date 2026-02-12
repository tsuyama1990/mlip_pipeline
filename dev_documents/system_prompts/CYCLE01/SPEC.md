# Cycle 01 Specification: Core Infrastructure & Mocking

## 1. Summary

Cycle 01 focuses on establishing the foundational infrastructure of the **PYACEMAKER** system. The primary objective is to create a robust, modular skeleton that can support the complex workflows of future cycles. This includes setting up the Python package structure, implementing the central **Orchestrator**, defining the **Configuration** schema using Pydantic, and establishing a unified **Logging** and **Error Handling** mechanism.

Crucially, this cycle will implement **Mock** versions of all five core components (Structure Generator, Oracle, Trainer, Dynamics Engine, Validator). These mocks will simulate the behavior of external physics codes (returning dummy structures, energies, or potentials) without requiring the actual binaries (LAMMPS, QE, etc.) to be installed. This allows us to verify the Orchestrator's logic and data flow immediately, enabling "Test-Driven Development" (TDD) for the rest of the project.

We will also deliver the Command Line Interface (CLI) entry point, `mlip-runner`, which will accept a configuration file and execute a (mock) workflow.

## 2. System Architecture

The following file structure will be created. Files in **bold** are to be created or significantly modified in this cycle.

```ascii
.
├── pyproject.toml              # Dependencies and Linter Config
├── README.md                   # Project Documentation
├── **src/**
│   └── **mlip_autopipec/**     # Main Package
│       ├── **__init__.py**
│       ├── **main.py**         # CLI Entrypoint (Typer)
│       ├── **constants.py**    # Global Constants
│       ├── **core/**           # Core Logic
│       │   ├── **__init__.py**
│       │   ├── **orchestrator.py**   # Main Controller (stub with mocks)
│       │   ├── **config_parser.py**  # YAML -> Pydantic
│       │   ├── **state_manager.py**  # Workflow State tracking
│       │   └── **logger.py**         # Centralized Logging
│       ├── **domain_models/**  # Pydantic Models
│       │   ├── **__init__.py**
│       │   ├── **config.py**         # Configuration Schema
│       │   ├── **datastructures.py** # Structure, Dataset, Potential
│       │   └── **enums.py**          # Enums (DFTCode, TaskType, etc.)
│       ├── **generator/**      # Structure Generator Module
│       │   ├── **__init__.py**
│       │   └── **interface.py**      # Base Class & Mock Implementation
│       ├── **oracle/**         # Oracle Module
│       │   ├── **__init__.py**
│       │   └── **interface.py**      # Base Class & Mock Implementation
│       ├── **trainer/**        # Trainer Module
│       │   ├── **__init__.py**
│       │   └── **interface.py**      # Base Class & Mock Implementation
│       ├── **dynamics/**       # Dynamics Module
│       │   ├── **__init__.py**
│       │   └── **interface.py**      # Base Class & Mock Implementation
│       └── **validator/**      # Validator Module
│           ├── **__init__.py**
│           └── **interface.py**      # Base Class & Mock Implementation
└── **tests/**
    ├── **conftest.py**         # Pytest fixtures
    ├── **unit/**               # Unit Tests
    │   ├── **test_config.py**
    │   ├── **test_orchestrator.py**
    │   └── **test_mocks.py**
    └── **uat/**                # User Acceptance Tests
        └── **verify_cycle01.py**
```

## 3. Design Architecture

The system is built on **Pydantic** for rigorous data validation and **Abstract Base Classes (ABC)** for component modularity.

### 3.1 Configuration Schema (`domain_models/config.py`)
The configuration is the single source of truth.
*   `GlobalConfig`: The root model.
*   `OrchestratorConfig`: Settings for the workflow (e.g., max_cycles, work_dir).
*   `ComponentConfig`: Polymorphic base model for component settings.
    *   `GeneratorConfig` (exploration strategies)
    *   `OracleConfig` (DFT code selection, resources)
    *   `TrainerConfig` (Pacemaker settings)
    *   `DynamicsConfig` (LAMMPS/EON settings)
    *   `ValidatorConfig` (Validation thresholds)

**Key Constraint**: All paths in the config must be validated to exist (or be creatable). Enums must be used for selection (e.g., `DFTCode.QUANTUM_ESPRESSO` instead of raw strings).

### 3.2 Component Interfaces (`*/interface.py`)
Each component (Generator, Oracle, etc.) must inherit from a common `BaseComponent` ABC.
*   **Methods**:
    *   `explore(context)` -> `List[Structure]` (Generator)
    *   `compute(structures)` -> `Dataset` (Oracle)
    *   `train(dataset)` -> `Potential` (Trainer)
    *   `simulate(potential, structure)` -> `Trajectory` (Dynamics)
    *   `validate(potential)` -> `ValidationResult` (Validator)

### 3.3 Mock Implementations
For Cycle 01, we implement `MockGenerator`, `MockOracle`, etc., which inherit from the interfaces.
*   `MockGenerator`: Returns random `ase.Atoms` objects.
*   `MockOracle`: Assigns random energies/forces to the atoms.
*   `MockTrainer`: Creates a dummy file named `potential.yace`.
*   `MockDynamics`: Returns a successful trajectory without running LAMMPS.
*   `MockValidator`: Returns `passed=True`.

This allows the `Orchestrator` to execute a full "Happy Path" loop without external binaries.

## 4. Implementation Approach

1.  **Project Skeleton**: Initialize the directory structure and `pyproject.toml`.
2.  **Domain Models**: Define the Pydantic models in `domain_models/`. This establishes the "Language" of the system.
3.  **Config Parser**: Implement `load_config` in `core/config_parser.py` to read YAML and validate against models.
4.  **Logging**: Setup `core/logger.py` using standard python logging, ensuring output to both console and file.
5.  **Component Interfaces**: Define the ABCs in `*/interface.py`.
6.  **Mock Components**: Implement the Mock versions in the same files (or dedicated `mock.py`).
7.  **Orchestrator**: Implement the main loop in `core/orchestrator.py` that instantiates components based on config and runs the cycle.
8.  **CLI**: Use `typer` to create `src/mlip_autopipec/main.py` with an `init` command (scaffold config) and `run` command.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Config Test**: Verify that valid YAMLs are parsed correctly and invalid ones raise `ValidationError`.
*   **Mock Test**: Verify that Mock components return data matching the expected types (e.g., `MockOracle` returns atoms with `calc` attached).
*   **Orchestrator Test**: Run the orchestrator with a mock config and verify it completes 1 cycle without error.

### 5.2 Integration Testing
*   **CLI Test**: Use `typer.testing.CliRunner` to invoke `mlip-runner run config.yaml` and check the exit code and stdout.

### 5.3 User Acceptance Testing (UAT)
*   **Scenario**: "Hello World of MLIP".
*   **Input**: A simple `config.yaml` specifying `execution_mode: mock`.
*   **Action**: User runs `mlip-runner run config.yaml`.
*   **Output**: The system prints logs indicating "Generation 0 completed", and produces a dummy `potential.yace` in the output directory.
