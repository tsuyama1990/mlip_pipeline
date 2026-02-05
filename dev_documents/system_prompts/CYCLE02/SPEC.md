# CYCLE 02 Specification: Trainer Module Integration

## 1. Summary

In this cycle, we replace the `MockTrainer` with a real **`PacemakerTrainer`**. This module interfaces with the `pacemaker` (ACE) library to perform actual model training. It also implements the "Delta Learning" strategy, where the model learns the difference between DFT forces/energy and a physics-based baseline (LJ or ZBL).

## 2. System Architecture

Files to be modified/created:

```ascii
src/mlip_autopipec/
├── config/
│   └── **training_config.py**      # Training specific settings
├── domain_models/
│   └── dataset.py              # Update for serialization
├── training/
│   ├── **__init__.py**
│   └── **pacemaker.py**        # Real Pacemaker Wrapper
└── orchestration/
    └── orchestrator.py         # Update to use PacemakerTrainer
```

## 3. Design Architecture

### 3.1. `TrainingConfig` (Pydantic)
*   Parameters for `pacemaker`:
    *   `cutoff`: float (default 5.0)
    *   `elements`: list[str] (e.g., ["Fe", "Pt"])
    *   `batch_size`: int
    *   `max_epochs`: int
    *   `physics_baseline`: str ("ZBL" or "LJ")

### 3.2. `PacemakerTrainer` Class
*   **Interface**: Implements `Trainer` protocol.
*   **Responsibilities**:
    1.  **Data Conversion**: Convert internal `Dataset` (list of ASE atoms) into Pacemaker's expected format (`.pckl.gzip` or `data.p7`).
    2.  **Config Generation**: Dynamically generate the `input.yaml` required by `pace_train`.
    3.  **Delta Learning**: If `physics_baseline="ZBL"` is set, configure the input YAML to use a hybrid potential definition.
    4.  **Execution**: Call `pace_train` via `subprocess`.
    5.  **Artifact Management**: Capture `potential.yace` and training logs.

## 4. Implementation Approach

1.  **Install Pacemaker**: Ensure `pacemaker` and `tensorpotential` are available in the dev environment (or mocked via subprocess if binary not present).
2.  **Config**: Add `TrainingConfig` to `GlobalConfig`.
3.  **Data Serialization**: Implement `Dataset.save_to_pacemaker_format()`.
4.  **Trainer Logic**:
    *   Create `src/mlip_autopipec/training/pacemaker.py`.
    *   Implement `train()` method.
    *   Use `yaml` library to write `input.yaml` for pacemaker.
    *   Use `subprocess.run(["pace_train", ...])`.
5.  **Orchestrator Update**: allow switching between `MockTrainer` and `PacemakerTrainer` via config.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_pacemaker_config.py`**: Verify that `PacemakerTrainer` generates a valid YAML string for `pace_train`, correctly including the ZBL/LJ sections.

### 5.2. Integration Testing
*   **`test_training_execution.py`**:
    *   Create a small dataset (2 structures).
    *   Run `PacemakerTrainer.train()`.
    *   **Mocking**: Since running actual training is slow/heavy, we will mock `subprocess.run` to return success and touch a dummy `potential.yace` file.
    *   **Real Mode**: If `pytest` is run with `--slow`, actually invoke `pace_train` (requires installed pacemaker).
