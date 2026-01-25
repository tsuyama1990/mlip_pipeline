# Cycle 01 Specification: Skeleton & Configuration Infrastructure

## 1. Summary

The primary objective of Cycle 01 is to establish the foundational infrastructure for the **PyAcemaker** project. This is the "Skeleton" phase where we define the project structure, set up the configuration management system, and implement the logging framework.

A robust configuration system is critical for PyAcemaker because the ultimate goal is a "Zero-Config" user experience where a single YAML file drives the entire complex workflow of DFT calculations, machine learning training, and MD simulations. We will use **Pydantic** to strictly validate this configuration file, ensuring that any user errors (e.g., missing paths, invalid thresholds) are caught immediately at startup rather than causing crashes hours into a simulation.

In this cycle, we will not yet run any physics simulations. Instead, we will build the "spinal cord" of the system: the `WorkflowManager` which reads the config, initializes the workspace (directories for data, potentials, logs), and prepares the state for the subsequent cycles. We will also establish the CLI entry point.

## 2. System Architecture

The focus of this cycle is on the `config`, `utils`, and the shell of `orchestration`.

### 2.1 File Structure

The following file structure will be implemented. **Bold** files are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── **__init__.py**
├── **main.py**                     # CLI Entry point using Typer or Argparse
├── **config/**                     # Configuration Module
│   ├── **__init__.py**
│   ├── **models.py**               # Aggregated Config Models
│   └── **schemas/**
│       ├── **__init__.py**
│       ├── **common.py**           # Common types (Path, Enums)
│       ├── **workflow.py**         # Workflow settings
│       ├── **dft.py**              # DFT (Oracle) settings
│       ├── **training.py**         # Training settings
│       ├── **inference.py**        # MD/EON settings
│       └── **validation.py**       # Validation settings
├── **orchestration/**
│   ├── **__init__.py**
│   ├── **manager.py**              # WorkflowManager (State Machine)
│   └── **state.py**                # WorkflowState (Data Class)
└── **utils/**
    ├── **__init__.py**
    └── **logging.py**              # Centralized logging setup
tests/
├── **conftest.py**
├── **test_config.py**              # Tests for Pydantic models
└── **test_manager.py**             # Tests for WorkflowManager
```

## 3. Design Architecture

We employ a **Schema-First Design** using Pydantic V2. This allows us to define the "contract" of our system explicitly.

### 3.1 Configuration Schemas (`src/mlip_autopipec/config/`)

We will define a hierarchy of configuration models.

*   **`WorkflowConfig`**:
    *   `project_name`: str
    *   `max_cycles`: int (default: 10)
    *   `work_dir`: Path (default: "./workspace")
    *   `continue_from`: Optional[int] (for resuming)

*   **`DFTConfig`**:
    *   `command`: str (e.g., "mpirun -np 4 pw.x")
    *   `pseudopotential_dir`: Path (Must exist)
    *   `kspacing`: float (default: 0.04)
    *   `scf_max_retries`: int

*   **`TrainingConfig`**:
    *   `potential_type`: Enum ("ace")
    *   `ace_cutoff`: float
    *   `baseline_type`: Enum ("lj", "zbl")
    *   `active_set_size`: int

*   **`InferenceConfig`**:
    *   `md_steps`: int
    *   `temperature`: float
    *   `uncertainty_threshold`: float (The $\gamma$ value)

*   **`ValidationConfig`**:
    *   `phonon_supercell`: List[int] (e.g., [4, 4, 4])
    *   `max_energy_rmse`: float

*   **`Config` (Root)**:
    *   Aggregates all above: `workflow`, `dft`, `training`, `inference`, `validation`.

### 3.2 Workflow State (`src/mlip_autopipec/orchestration/state.py`)

We need a way to track the *dynamic* state of the system as it progresses through cycles.
*   **`WorkflowState`**:
    *   `current_cycle`: int
    *   `latest_potential_path`: Optional[Path]
    *   `dataset_path`: Optional[Path]
    *   `status`: Enum ("IDLE", "RUNNING", "HALTED", "ERROR")

### 3.3 Workflow Manager (`src/mlip_autopipec/orchestration/manager.py`)

This class is the heart of the automation.
*   **Responsibilities**:
    1.  Load and validate `Config`.
    2.  Initialize the directory structure (create `work_dir`, `logs/`, `potentials/`).
    3.  Setup logging via `utils.logging`.
    4.  Maintain `WorkflowState`.
    5.  Provide a `run()` method (which will be empty for now, just logging "Starting...").

### 3.4 Logging (`src/mlip_autopipec/utils/logging.py`)

*   Use Python's standard `logging` library.
*   Configure a console handler (INFO level) and a file handler (DEBUG level).
*   Format: `[TIMESTAMP] [LEVEL] [MODULE] Message`.

## 4. Implementation Approach

1.  **Step 1: Define Common Types.**
    *   Create `config/schemas/common.py` to define shared Enums and custom types (e.g., a validator that checks if a path exists).

2.  **Step 2: Implement Component Configs.**
    *   Implement `dft.py`, `training.py`, etc., one by one.
    *   Add Pydantic validators (e.g., ensure `kspacing > 0`, `ace_cutoff > 0`).

3.  **Step 3: Implement Root Config.**
    *   Create `models.py` to aggregate them into the main `Config` object.
    *   Implement a `load_config(path: Path)` function that reads YAML and returns a `Config` instance.

4.  **Step 4: Implement Logging.**
    *   Create `utils/logging.py` with a `setup_logging` function.

5.  **Step 5: Implement WorkflowManager.**
    *   Create the class structure.
    *   Implement `__init__` to accept a config path.
    *   Implement `_setup_workspace()` to create directories.

6.  **Step 6: CLI Entry Point.**
    *   Create `main.py`. Use `argparse` to accept `--config`.
    *   Instantiate `WorkflowManager` and call `run()`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Config Validation Tests (`test_config.py`):**
    *   **Valid Config:** Create a minimal valid YAML and assert it loads correctly.
    *   **Missing Fields:** Assert that missing required fields raise `ValidationError`.
    *   **Invalid Types:** Assert that passing a string for `max_cycles` (if not castable) fails.
    *   **Logic Checks:** Assert that `uncertainty_threshold` cannot be negative.

*   **State Tests:**
    *   Verify `WorkflowState` initializes with default values (cycle=0).

### 5.2 Integration Testing
*   **Manager Initialization (`test_manager.py`):**
    *   Test that `WorkflowManager("config.yaml")` successfully creates the directory tree (`workspace/logs`, `workspace/potentials`) on the filesystem.
    *   Test that the log file is created and written to.

*   **CLI Test:**
    *   Invoke `python -m mlip_autopipec.main --config test_config.yaml` using `subprocess` and check for exit code 0.
