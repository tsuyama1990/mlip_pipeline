# Cycle 01 Specification: Core Infrastructure & MACE Integration

## 1. Summary
The objective of Cycle 01 is to establish the foundational infrastructure for the **PYACEMAKER** system. This includes setting up the project structure, implementing the configuration management system using Pydantic, and defining the core interfaces (Abstract Base Classes) for the modular components.

A critical part of this cycle is the initial integration of the **MACE** (Machine Learning Interatomic Potentials) framework. We will implement the `MaceSurrogateOracle` class, which serves as the interface to the MACE foundation models. Although full active learning and fine-tuning logic will be added in later cycles, Cycle 01 must ensure that the system can successfully load a MACE model (or a mock version for testing) and perform basic energy/force predictions.

By the end of this cycle, we should have a functional `Orchestrator` that can read a `config.yaml` file, validate the settings, and initialize the `MaceSurrogateOracle`.

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified in this cycle.

```text
src/pyacemaker/
├── __init__.py
├── **main.py**               # Entry point (CLI)
├── **orchestrator.py**       # Main controller class
├── core/
│   ├── **__init__.py**
│   ├── **base.py**           # BaseModule Definition
│   ├── **interfaces.py**     # Abstract Base Classes (BaseOracle, BaseTrainer)
│   ├── **config.py**         # Pydantic Configuration Models (MaceConfig, OracleConfig)
│   └── **exceptions.py**     # Custom exceptions
├── domain_models/
│   ├── **__init__.py**
│   └── **structure.py**      # StructureMetadata (Atoms + Metadata)
├── modules/
│   ├── **oracle.py**         # MaceSurrogateOracle & DFTOracle (Wrappers)
└── oracle/
    ├── **__init__.py**
    ├── **manager.py**        # DFTManager (Existing)
    └── **mace_manager.py**   # MACE Manager Implementation (New)
```

## 3. Design Architecture

### 3.1. Configuration Management (`core/config.py`)
We will use **Pydantic** to define the schema for `config.yaml`. This ensures strict type checking and validation at runtime.

-   **`MaceConfig`**: Settings for MACE model path, device (CPU/GPU), precision, and batch size.
-   **`OracleConfig`**: Contains `dft` (DFTConfig) and `mace` (MaceConfig).
-   **`SystemConfig`**: Target elements (e.g., ["Fe", "Pt"]) and simulation box parameters.
-   **`PYACEMAKERConfig`**: Top-level configuration aggregating the above.

### 3.2. Domain Models (`domain_models/structure.py`)
To ensure type safety across modules, we define a `StructureMetadata` class (wrapping `ase.Atoms` logic) that carries essential metadata.

-   **`StructureMetadata`**:
    -   `features`: dict (contains `atoms` as `ase.Atoms`).
    -   `energy`: Optional[float].
    -   `forces`: Optional[list[list[float]]].
    -   `uncertainty_state`: Optional[UncertaintyState] (for Active Learning).
    -   `source`: str (e.g., "DIRECT", "MD", "DFT").

### 3.3. Core Interfaces (`core/interfaces.py`)
All modules must inherit from abstract base classes to ensure consistency.

-   **`BaseModule`** (in `core/base.py`): Common methods for logging and configuration access.
-   **`BaseOracle`** (in `modules/oracle.py`):
    -   `compute_batch(structures: Iterable[StructureMetadata]) -> Iterator[StructureMetadata]`: Batch prediction.

### 3.4. MACE Oracle (`modules/oracle.py` & `oracle/mace_manager.py`)
This class implements `BaseOracle` using the `mace-torch` library via a helper manager.
-   **Key Responsibility**: Load the MACE model from a file or URL (e.g., MACE-MP-0).
-   **Mock Mode**: If `config.oracle.mock` is True, it should load a dummy model (via `MockOracle` or internal check) that returns random (but consistent) energy/forces to avoid large dependencies during CI/CD.

## 4. Implementation Approach

1.  **Project Setup**: Initialize `src/pyacemaker` and subdirectories (mostly done).
2.  **Define Domain Models**: Verify `StructureMetadata` in `domain_models/structure.py` (Done).
3.  **Implement Config**: Update `OracleConfig` and add `MaceConfig` in `core/config.py`.
4.  **Create Base Classes**: Verify `BaseOracle` in `modules/oracle.py` (Done).
5.  **Implement MACE Oracle**:
    -   Create `MaceManager` in `oracle/mace_manager.py` to handle `mace-torch` interaction.
    -   Implement `load_model` method.
    -   Implement `compute` method using `mace.calculators.MACECalculator`.
    -   Update `MaceSurrogateOracle` in `modules/oracle.py`.
6.  **Develop Orchestrator**:
    -   Update `Orchestrator` in `orchestrator.py`.
    -   Add `__init__` that takes a config file path.
    -   Implement logic to parse config and instantiate `MaceSurrogateOracle`.
7.  **CLI Entry Point**: Create `main.py` to expose the orchestrator via command line (e.g., `python -m pyacemaker.main config.yaml`).

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Validation**: Create valid and invalid `config.yaml` files. Assert that `PYACEMAKERConfig` parses valid ones and raises `ValidationError` for invalid ones (e.g., missing elements).
-   **Domain Models**: Verify `StructureMetadata` correctly wraps `ase.Atoms` and validates array shapes (e.g., forces match atom count).

### 5.2. Integration Testing (Mocked)
-   **MACE Loading**: Test `MaceSurrogateOracle` initialization in "Mock Mode". Verify it does not crash and returns a result.
-   **Orchestrator Init**: Initialize `Orchestrator` with a valid config. Verify that `orchestrator.oracle` is an instance of `MaceSurrogateOracle`.

### 5.3. Manual Verification
-   Run `python -m pyacemaker.main valid_config.yaml` and check logs to ensure successful initialization and "Ready" status.
