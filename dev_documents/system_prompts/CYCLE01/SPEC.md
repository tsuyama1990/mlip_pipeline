# Cycle 01 Specification: Architecture Skeleton & Configuration

## 1. Summary

Cycle 01 focuses on laying the foundational infrastructure for the PyAcemaker system. The primary goal is to establish a robust, modular project structure and a type-safe configuration management system using Pydantic. This cycle will not involve running any physical simulations (DFT or MD) but will ensure that the "Brain" of the system—the Orchestrator—can be initialised, read user inputs, validate them, and set up the execution environment. We will also implement a centralized logging system to track the system's state, which is crucial for the autonomous nature of the project.

By the end of this cycle, we will have a CLI entry point that can accept a `config.yaml`, validate its contents against strict schemas, and instantiate the Orchestrator class. This "Skeleton" will serve as the backbone for connecting the functional modules (Oracle, Trainer, etc.) in subsequent cycles.

## 2. System Architecture

The file structure is designed to enforce separation of concerns. We introduce the `src/mlip_autopipec` package structure.

### File Structure

**Files to be created in this cycle are marked in Bold.**

```ascii
src/
└── **mlip_autopipec**/
    ├── **__init__.py**
    ├── **app.py**                  # CLI Entry Point
    ├── **orchestrator.py**         # Main Controller Class (Skeleton)
    ├── **exceptions.py**           # Custom Exception Classes
    ├── **config**/
    │   ├── **__init__.py**
    │   ├── **base.py**             # Base Pydantic Models
    │   ├── **dft.py**              # DFT Configuration Schema
    │   ├── **training.py**         # Training Configuration Schema
    │   ├── **inference.py**        # Inference/MD Configuration Schema
    │   ├── **validation.py**       # Validation Configuration Schema
    │   └── **main.py**             # Global Config Aggregator
    └── **utils**/
        ├── **__init__.py**
        ├── **logging.py**          # Centralized Logging Setup
        └── **paths.py**            # Path Management Utilities
tests/
├── **test_config.py**              # Tests for Configuration Validation
└── **test_orchestrator_init.py**   # Tests for Orchestrator Initialization
```

## 3. Design Architecture

The core of this cycle is the Configuration Design. We adhere to "Schema-First Development".

### Pydantic Models

We define a hierarchy of configuration models. This ensures that invalid user inputs are caught immediately at startup with clear error messages.

1.  **`GlobalConfig` (in `config/main.py`)**:
    *   The root model.
    *   Fields: `project_name` (str), `work_dir` (Path), `dft` (DFTConfig), `training` (TrainingConfig), `inference` (InferenceConfig), `validation` (ValidationConfig).
    *   **Invariant**: `work_dir` must be writable.

2.  **`DFTConfig` (in `config/dft.py`)**:
    *   Parameters for the Oracle.
    *   Fields: `qe_command` (str), `pseudopotential_dir` (Path), `kspacing` (float), `scf_limit` (int).
    *   **Constraint**: `kspacing` must be positive. `pseudopotential_dir` must exist.

3.  **`TrainingConfig` (in `config/training.py`)**:
    *   Parameters for Pacemaker.
    *   Fields: `potential_type` (Literal["ace"]), `r_cut` (float), `max_degree` (int).
    *   **Constraint**: `r_cut` must be within physically reasonable bounds (e.g., 2.0 to 10.0 Angstroms).

4.  **`InferenceConfig` (in `config/inference.py`)**:
    *   Parameters for LAMMPS/EON.
    *   Fields: `engine` (Literal["lammps", "eon"]), `temperature` (float), `md_steps` (int), `uncertainty_threshold` (float).

5.  **`ValidationConfig` (in `config/validation.py`)**:
    *   Fields: `phonon_check` (bool), `elastic_check` (bool), `max_force_rmse` (float).

### Orchestrator Logic

The `Orchestrator` class (in `orchestrator.py`) will implement the Singleton pattern (or effectively act as one per run).
*   **`__init__(self, config: GlobalConfig)`**: Stores the config, sets up logging, and prepares the workspace directories.
*   **`run(self)`**: The main entry point. For Cycle 01, this will simply log "System Initialized" and exit.

### Logging System

*   implemented in `utils/logging.py`.
*   Uses Python's built-in `logging` module.
*   Configures both file output (`system.log`) and console output.
*   Console output should be concise (INFO level), while file output should be verbose (DEBUG level).

## 4. Implementation Approach

1.  **Setup Package**: Create the directory structure `src/mlip_autopipec/...`.
2.  **Implement Utilities**: Write `utils/logging.py` first, as other modules will depend on it.
3.  **Implement Config Models**:
    *   Start with leaf configs (`DFTConfig`, etc.).
    *   Implement the root `GlobalConfig`.
    *   Add Pydantic validators (e.g., `@field_validator`) to check for file existence or value ranges.
4.  **Implement Orchestrator**:
    *   Create the class.
    *   Import the config models.
    *   Implement directory creation logic in `__init__`.
5.  **Implement CLI**:
    *   Create `app.py`.
    *   Use `argparse` or `typer` to accept `--config`.
    *   Load the YAML file, parse it into `GlobalConfig`, and pass it to `Orchestrator`.

## 5. Test Strategy

### Unit Testing Approach
*   **Config Validation**: Create a series of valid and invalid YAML dictionaries.
    *   Assert that `GlobalConfig(**valid_dict)` succeeds.
    *   Assert that `GlobalConfig(**invalid_dict)` raises `ValidationError`.
    *   Specifically test edge cases: negative temperatures, missing paths, empty strings.
*   **Orchestrator Init**:
    *   Mock the filesystem (using `tmp_path` fixture).
    *   Initialize `Orchestrator` and verify that it creates the required `work_dir` and subdirectories (`logs/`, `data/`).

### Integration Testing Approach
*   **CLI Test**:
    *   Simulate a command-line call to `app.py` with a sample `config.yaml`.
    *   Verify that the application runs and exits with code 0.
    *   Verify that the log file is created and contains the startup message.
