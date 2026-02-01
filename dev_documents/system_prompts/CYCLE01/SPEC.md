# Specification: Cycle 01 - Foundation & Basic Loop

## 1. Summary

Cycle 01 is the "Genesis" phase of the **PYACEMAKER** project. The primary objective is to establish the structural and architectural foundation upon which the entire active learning pipeline will be built. In this cycle, we will not yet connect to external physics engines (Quantum Espresso or LAMMPS) in a production capacity. Instead, we will focus on creating the "Orchestrator" — the central nervous system of the software — and the "Configuration Manager".

The success of this cycle is defined by the system's ability to read a user-provided `config.yaml` file, validate its contents using strict Pydantic schemas, and execute a "Skeleton Loop". This loop will simulate the phases of Exploration, Calculation, and Training using mock objects, but will perform a *real* execution of the `pace_train` command (wrapped in a class) if the environment permits, or mock it otherwise.

This cycle lays the groundwork for:
1.  **Robust Error Handling**: By validating inputs at startup.
2.  **State Management**: By tracking the iteration count and workflow status.
3.  **Extensibility**: By defining the interfaces (Protocols) for future modules.

## 2. System Architecture

This section provides the exact code blueprints for the file structure and key components to be implemented in Cycle 01.

### 2.1 File Structure

The following ASCII tree depicts the files to be created (marked in **bold**) or modified.

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── __init__.py
│       ├── **main.py**                     # Entry point (CLI)
│       ├── **logging_config.py**           # Logger setup
│       ├── config/
│       │   ├── __init__.py
│       │   └── **config_model.py**         # Pydantic schemas for config.yaml
│       ├── domain_models/
│       │   ├── __init__.py
│       │   ├── **workflow.py**             # WorkflowState definition
│       │   └── **potential.py**            # Potential artifact definition
│       ├── orchestration/
│       │   ├── __init__.py
│       │   ├── **orchestrator.py**         # Main loop logic
│       │   └── **state.py**                # State persistence logic
│       ├── physics/
│       │   ├── __init__.py
│       │   └── training/
│       │       ├── __init__.py
│       │       └── **pacemaker.py**        # Wrapper for pace_train
│       └── utils/
│           ├── __init__.py
│           └── **file_ops.py**             # Safe file operations
├── tests/
│   ├── **conftest.py**
│   ├── unit/
│   │   ├── **test_config.py**
│   │   └── **test_orchestrator.py**
│   └── integration/
│       └── **test_skeleton_loop.py**
└── **config.yaml** (Example)
```

### 2.2 Component Blueprints

#### `src/mlip_autopipec/config/config_model.py`
This module defines the strict schema for user input.

```python
from pydantic import BaseModel, Field, FilePath, DirectoryPath
from typing import Optional, List

class ProjectConfig(BaseModel):
    name: str
    seed: int = 42

class TrainingConfig(BaseModel):
    dataset_path: FilePath
    max_epochs: int = 100
    command: str = "pace_train"

# ... (Other configs for future cycles, kept minimal for now)

class Config(BaseModel):
    project: ProjectConfig
    training: TrainingConfig
    # ...
```

#### `src/mlip_autopipec/orchestration/orchestrator.py`
The brain of the operation.

```python
class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.state = WorkflowState()
        self.trainer = PacemakerTrainer(config.training)

    def run(self):
        """Executes the Active Learning Cycle."""
        while self.state.iteration < self.config.orchestrator.max_iterations:
            logging.info(f"Starting Cycle {self.state.iteration}")

            # Phase 1: Exploration (Mock for Cycle 01)
            self._run_exploration()

            # Phase 2: Selection & Calculation (Mock for Cycle 01)
            self._run_oracle()

            # Phase 3: Training (Real implementation)
            potential_path = self.trainer.train(
                dataset=self.config.training.dataset_path,
                previous_potential=self.state.current_potential
            )

            self.state.update_potential(potential_path)
            self.state.save()
```

#### `src/mlip_autopipec/physics/training/pacemaker.py`
Wrapper for the external tool.

```python
import subprocess
from pathlib import Path

class PacemakerTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, dataset: Path, previous_potential: Path | None) -> Path:
        """Runs pace_train via subprocess."""
        cmd = [self.config.command, "--dataset", str(dataset)]
        # ... logic to build command ...
        subprocess.run(cmd, check=True)
        return Path("output_potential.yace")
```

## 3. Design Architecture

The design philosophy for Cycle 01 is **"Configuration as Code"** and **"Fail Fast"**.

### 3.1 Pydantic-Based Configuration
The `Config` object is not just a dictionary; it is a strictly typed object graph.
*   **Validation**: File paths (`FilePath`, `DirectoryPath`) are checked for existence at startup. If the dataset file is missing, the program crashes immediately with a clear error message, rather than failing 2 hours later.
*   **Defaults**: Sensible defaults are provided for optional parameters (e.g., `max_epochs=100`), reducing the burden on the user.
*   **Immutability**: Once loaded, the config object should be treated as immutable to prevent runtime side effects.

### 3.2 State Management (`WorkflowState`)
The `WorkflowState` model (in `domain_models/workflow.py`) is designed to support **Idempotency**.
*   **Fields**:
    *   `iteration` (int): Current cycle number.
    *   `current_potential_path` (Optional[Path]): Path to the latest accepted potential.
    *   `history` (List[Dict]): A log of metrics from previous cycles.
*   **Persistence**: The state is serialized to `state.json` at the end of every critical step. If the process is killed (e.g., by a scheduler timeout), restarting it will reload `state.json` and resume from the last successful checkpoint.

### 3.3 Domain Models
*   **`Potential`**: Represents the artifact. It encapsulates the file path but also metadata like its "Generation ID". This abstraction allows us to swap the underlying file format (e.g., `.yace` vs `.pot`) in the future without breaking the Orchestrator logic.

## 4. Implementation Approach

The implementation will follow a linear path, building from the bottom up.

### Step 1: Project Setup & Logging
1.  Initialize the git repository (if not already done).
2.  Create the directory structure defined in System Architecture.
3.  Implement `logging_config.py` to ensure all modules output logs with timestamps and severity levels. This is crucial for debugging the "Silent" failures often seen in long-running jobs.

### Step 2: Configuration Module
1.  Install `pydantic`.
2.  Define `ProjectConfig` and `TrainingConfig` models.
3.  Create a `load_config(path)` function that parses a YAML file and returns a `Config` instance.
4.  **Verification**: Create a test `test_config.py` that tries to load a valid yaml and an invalid yaml (missing fields), asserting that the correct exceptions are raised.

### Step 3: Domain Models & State
1.  Define `WorkflowState` Pydantic model.
2.  Implement `StateManager` class in `state.py` that handles atomic writing of the JSON file (write to `.tmp` then rename) to prevent corruption during crashes.

### Step 4: Trainer Adapter
1.  Implement `PacemakerTrainer`.
2.  Use `subprocess.run` to call `pace_train`.
3.  **Crucial**: Since `pace_train` might not be installed on the dev machine, implement a "Dry Run" or "Mock" mode where the class just `touch`es a dummy output file if a specific flag is set.

### Step 5: The Orchestrator
1.  Assemble the pieces. `Orchestrator` initializes `Config`, `StateManager`, and `Trainer`.
2.  Write the `run()` loop.
3.  For Cycle 01, the "Exploration" and "Oracle" phases will be placeholder methods that simply log "Phase X completed".

### Step 6: CLI Entry Point
1.  Implement `main.py`.
2.  Use `argparse` to accept the config file path.
3.  Instantiate and run the Orchestrator.

## 5. Test Strategy

### 5.1 Unit Testing Approach (Min 300 words)
Unit tests will focus on the correctness of the internal logic, isolated from external systems.
*   **Config Testing**: We will create a suite of YAML snippets representing various edge cases (e.g., negative epochs, non-existent paths). The tests must verify that Pydantic raises `ValidationError` with informative messages.
*   **State Persistence**: We will test the `StateManager` by initializing it, modifying the state, saving it, and then loading it back from disk in a fresh instance. We will specifically test the atomic write operation to ensure no partial files are created.
*   **Trainer Logic**: We will test `PacemakerTrainer` by mocking `subprocess.run`. We verify that the constructed command list (list of strings) is exactly what we expect (e.g., `["pace_train", "--dataset", "data.pckl", ...]`). This ensures we are calling the external tool correctly without actually needing the tool installed.

### 5.2 Integration Testing Approach (Min 300 words)
Integration tests will verify that the components work together as a system.
*   **The "Skeleton Loop" Test**: We will create a test fixture that sets up a temporary directory with a valid `config.yaml` and a dummy `dataset.pckl`. We will then run the `Orchestrator.run()` method (with max_iterations=1).
*   **Success Criteria**:
    1.  The code runs without throwing unhandled exceptions.
    2.  A `state.json` file is created and contains `iteration: 1`.
    3.  A log file is generated containing expected messages ("Starting Cycle 0", "Training completed").
    4.  (If using Mock Trainer) A dummy `output.yace` file is present in the expected output directory.
*   **Environment**: These tests must run in the CI environment. Since the CI might not have `pace_train`, the integration test must seamlessly default to the "Mock Trainer" implementation, perhaps triggered by an environment variable `PYACEMAKER_MOCK_MODE=1`.
