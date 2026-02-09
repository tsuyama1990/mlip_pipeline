# Cycle 01: Core Framework & Infrastructure Specification

## 1. Summary

The primary objective of Cycle 01 is to establish the foundational architecture of the PyAceMaker system. This involves setting up the project directory structure, implementing the configuration management system using Pydantic, and defining the core interfaces (Abstract Base Classes) that all subsequent components will adhere to. A robust logging system and a dependency injection mechanism (Component Factory) will also be implemented. This cycle lays the groundwork for the modular "Active Learning" pipeline, ensuring that future components (Oracle, Trainer, etc.) can be plugged in seamlessly.

## 2. System Architecture

This cycle focuses on the `core` and `domain_models` modules, as well as the skeleton of the `components` package.

### File Structure

The following file structure will be created. **Bold** files are to be implemented in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`__init__.py`**
        *   **`main.py`** (Entry point)
        *   **`config.py`** (Configuration Pydantic Models)
        *   **`factory.py`** (Component Factory)
        *   **`constants.py`** (System Constants)
        *   **`core/`**
            *   **`__init__.py`**
            *   **`orchestrator.py`** (Orchestrator Shell)
            *   **`logger.py`** (Logging setup)
        *   **`components/`**
            *   **`__init__.py`**
            *   **`base.py`** (Abstract Base Classes)
        *   **`domain_models/`**
            *   **`__init__.py`**
            *   **`enums.py`** (System Enums)
*   **`tests/`**
    *   **`unit/`**
        *   **`test_config.py`**
        *   **`test_factory.py`**

## 3. Design Architecture

### 3.1 Domain Models & Configuration

The system uses Pydantic V2 for strict configuration validation.

#### `config.py`
This file defines the schema for the `config.yaml` file.
*   **`BaseComponentConfig`**: Base model for all component configs.
*   **`OrchestratorConfig`**: Configuration for the main loop (e.g., number of cycles, work directory).
*   **`ExperimentConfig`**: Root configuration model containing nested configs for all components.

#### `enums.py`
Defines string enums for type safety.
*   **`ComponentRole`**: `GENERATOR`, `ORACLE`, `TRAINER`, `DYNAMICS`, `VALIDATOR`.
*   **`OracleType`**: `QE`, `VASP`, `MOCK`.
*   **`GeneratorType`**: `RANDOM`, `ADAPTIVE`.

### 3.2 Core Components

#### `base.py` (Abstract Base Classes)
Defines the contract for each component role.
*   **`BaseGenerator`**: Must implement `generate(n_structures: int) -> list[Structure]`.
*   **`BaseOracle`**: Must implement `compute(structures: list[Structure]) -> list[CalculationResult]`.
*   **`BaseTrainer`**: Must implement `train(dataset: list[CalculationResult]) -> PotentialArtifact`.
*   **`BaseDynamics`**: Must implement `explore(potential: PotentialArtifact) -> ExplorationResult`.

#### `factory.py` (Dependency Injection)
*   **`ComponentFactory`**: A registry-based factory that instantiates concrete classes based on the configuration (e.g., returns `QEOracle` if config says `type: "QE"`).

#### `orchestrator.py`
*   **`Orchestrator`**: The main class that initializes components via the Factory and manages the execution flow. In Cycle 01, it will verify component instantiation but will not run the loop yet.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `src/mlip_autopipec` and `tests` directories.
2.  **Constants & Enums**: Define system-wide constants (e.g., default paths) and enums in `constants.py` and `domain_models/enums.py`.
3.  **Base Classes**: Implement the ABCs in `components/base.py`. Use `abc.ABC` and `@abstractmethod`.
4.  **Configuration**: Implement Pydantic models in `config.py`. Ensure strict validation (e.g., `forbid` extra fields).
5.  **Logging**: Implement structured logging in `core/logger.py` using the standard `logging` library with a custom formatter.
6.  **Factory**: Implement `ComponentFactory` in `factory.py`. Use a dictionary registry pattern.
7.  **Orchestrator**: Implement the `__init__` method of `Orchestrator` to load config and instantiate components.
8.  **Main Entry**: Create `main.py` to parse CLI arguments (path to config file) and instantiate the Orchestrator.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_config.py`**:
    *   Create valid and invalid YAML/Dictionary inputs.
    *   Assert that `ExperimentConfig.model_validate()` raises `ValidationError` for missing required fields or invalid types.
    *   Verify default values are correctly populated.
*   **`test_factory.py`**:
    *   Register a mock component class.
    *   Verify that `ComponentFactory.create()` returns an instance of the correct class.
    *   Verify that passing an unknown type raises `ValueError`.

### 5.2 Integration Testing
*   **CLI Test**:
    *   Run `python -m mlip_autopipec.main --config=dummy_config.yaml`.
    *   Verify that the application starts, logs the startup message, and exits gracefully (or crashes with a specific config error if the file is missing).
