# Cycle 01 Specification: Core Infrastructure & Mocks

## 1. Summary
This cycle establishes the foundational architecture of the PYACEMAKER project. The primary goal is to create a robust skeleton that defines the contracts (interfaces) for all future components and provides "Mock" implementations to enable immediate end-to-end testing of the orchestration logic. We will also implement the configuration management system using Pydantic and the Command Line Interface (CLI) using Typer. By the end of this cycle, we will be able to run a "Hello World" version of the pipeline that reads a configuration file, instantiates mock components, and logs their activity without performing any heavy physical calculations.

## 2. System Architecture

The following file structure will be created. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/
└── mlip_autopipec/
    ├── **__init__.py**
    ├── **main.py**                 # CLI Entry Point
    ├── **factory.py**              # Component Factory (Dependency Injection)
    ├── **domain_models/**          # Data Transfer Objects
    │   ├── **__init__.py**
    │   ├── **structure.py**        # Structure Model
    │   ├── **potential.py**        # Potential Model
    │   ├── **validation.py**       # Validation Result Model
    │   └── **config.py**           # Global Configuration (Pydantic)
    ├── **interfaces/**             # Abstract Base Classes
    │   ├── **__init__.py**
    │   ├── **orchestrator.py**
    │   ├── **generator.py**
    │   ├── **oracle.py**
    │   ├── **trainer.py**
    │   ├── **dynamics.py**
    │   └── **validator.py**
    ├── **infrastructure/**         # Concrete Implementations (Mocks)
    │   ├── **__init__.py**
    │   └── **mocks.py**            # All Mock Classes
    └── **utils/**
        ├── **__init__.py**
        └── **logging.py**          # Logging Setup
```

## 3. Design Architecture

### 3.1. Domain Models
We will leverage Pydantic to ensure strict type safety and validation across the system.
-   **`Structure`**: Represents an atomic configuration. It must support serialisation to/from dictionaries to allow saving the state of the pipeline. Key fields: `positions` (Nx3 float array), `atomic_numbers` (N int array), `cell` (3x3 float array), `pbc` (3 bool array).
-   **`GlobalConfig`**: The root configuration object. It uses **Discriminated Unions** to allow polymorphic configuration. For example, `oracle: OracleConfig` can be either `MockOracleConfig` or `QuantumEspressoConfig`. This is critical for the "Zero-Config" philosophy.

### 3.2. Interfaces (ABCs)
All major components must inherit from Abstract Base Classes (ABCs). This enforces a consistent API and facilitates the "Strategy Pattern".
-   `BaseOracle.compute(structures: list[Structure]) -> list[Structure]`: Returns structures with energy/forces.
-   `BaseTrainer.train(dataset: list[Structure]) -> Potential`: Returns a path to the trained potential.
-   `BaseDynamics.run(potential: Potential, ...) -> ExplorationResult`: Returns new structures or halt conditions.

### 3.3. Dependency Injection (Factory)
A `create_component(config: ComponentConfig)` factory function will be implemented in `factory.py`. This decouples the instantiation logic from the business logic, allowing the Orchestrator to remain agnostic of whether it is using a Mock or a Real component.

## 4. Implementation Approach

### Step 1: Domain Models
Define the Pydantic models in `domain_models/`. Ensure `Structure` can handle numpy arrays (via custom validators or lists). Define the `GlobalConfig` structure with support for nested component configs.

### Step 2: Interfaces
Define the ABCs in `interfaces/`. Use `abc.ABC` and `@abstractmethod`. Define the method signatures carefully, ensuring they rely only on Domain Models.

### Step 3: Mock Implementations
Implement `MockOracle`, `MockTrainer`, `MockDynamics`, etc., in `infrastructure/mocks.py`.
-   `MockOracle`: Adds random noise to forces and energy of input structures.
-   `MockTrainer`: Creates a dummy file (e.g., `dummy.yace`) and returns its path.
-   `MockDynamics`: Returns a random "Halt" or "Converged" status to simulate active learning decisions.

### Step 4: CLI & Logging
Implement `main.py` using `typer`.
-   Command: `run <config_path>`
-   Action: Load config, setup logging, instantiate Orchestrator (mock for now), and run.
-   Logging: Configure `structlog` or standard `logging` to write to both console and a file (`mlip.log`).

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Parsing**: Create valid and invalid YAML files. assert that `GlobalConfig.from_yaml` correctly parses valid files and raises `ValidationError` for invalid ones.
-   **Mock Behaviour**: Instantiate `MockOracle` and call `compute`. Assert that the returned structures have `energy` and `forces` properties populated.
-   **Factory**: specific test to ensure `create_oracle(MockOracleConfig)` returns an instance of `MockOracle`.

### 5.2. Integration Testing (CLI Smoke Test)
-   Use `typer.testing.CliRunner`.
-   Run `mlip-pipeline --help` and assert exit code 0.
-   Run `mlip-pipeline run invalid_config.yaml` and assert exit code != 0 (graceful failure).
