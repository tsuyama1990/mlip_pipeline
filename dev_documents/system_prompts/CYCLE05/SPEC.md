# Cycle 05 Specification: Validation Module

## 1. Summary
This cycle implements the **Validator** component, the quality assurance gatekeeper of the system. Its role is to assess the performance of the trained MLIP against the ground-truth test set and physical constraints. It calculates standard metrics (RMSE for Energy, Force, Stress) and performs a "Crash Test" (short MD run) to ensure the potential is numerically stable. If a potential fails validation, the Orchestrator will be notified to either stop or take corrective action (though robust recovery is a Cycle 7 feature).

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**     # [MODIFY] Add ValidationConfig
│       ├── domain_models/
│       │   └── **validation.py**       # [NEW] ValidationResult model
│       ├── infrastructure/
│       │   ├── **validator/**
│       │   │   ├── **__init__.py**
│       │   │   ├── **adapter.py**      # [NEW] Validator implementation
│       │   │   └── **metrics.py**      # [NEW] RMSE calculation logic
└── tests/
    └── unit/
        └── **test_validator.py**       # [NEW] Tests for Validator
```

## 3. Design Architecture

### 3.1. `ValidationConfig` (Pydantic)
*   `test_set_ratio`: float (e.g., 0.1).
*   `energy_rmse_threshold`: float (meV/atom).
*   `force_rmse_threshold`: float (eV/A).
*   `stability_test_steps`: int (Number of MD steps for crash test).

### 3.2. `ValidationResult` (Pydantic)
*   `passed`: bool.
*   `metrics`: Dict[str, float] (e.g., `{'energy_rmse': 2.5, 'force_rmse': 0.04}`).
*   `artifacts`: List[Path] (Paths to plots/reports).
*   `reason`: Optional[str] (Failure reason).

### 3.3. `Validator` Class
Implements `BaseValidator`.
*   **Responsibilities**:
    1.  `validate(potential, test_set)`: Main entry point.
    2.  `calculate_metrics()`: Compute RMSE/MAE.
    3.  `run_stability_test()`: Run a short MD simulation (using `LammpsDynamics` or a simplified internal check) to see if atoms fly apart.
    4.  `generate_report()`: Create a summary JSON or HTML.

## 4. Implementation Approach

1.  **Metrics Logic**: Implement `infrastructure/validator/metrics.py`.
    *   Functions to compute RMSE between two lists of `Atoms` (predicted vs reference).
    *   `evaluate(model, dataset)`: Uses `pacemaker` or `lammps` to predict forces for the test set, then compares with DFT labels.
2.  **Implement Adapter**: Create `infrastructure/validator/adapter.py`.
    *   Load the Test Set (held out from training).
    *   Run predictions.
    *   Compare against thresholds in `ValidationConfig`.
    *   Return `ValidationResult`.
3.  **Integration**: Update `Orchestrator` to call `validator.validate()` after training.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Metrics Calculation**:
    *   Create two sets of Atoms with known differences.
    *   Call `calculate_rmse`.
    *   **Assert**: The result matches the manual calculation.
*   **Threshold Logic**:
    *   Mock the metrics return values.
    *   Call `validate()` with strict thresholds.
    *   **Assert**: Returns `passed=False`.
    *   Call with loose thresholds.
    *   **Assert**: Returns `passed=True`.

### 5.2. Integration Testing
*   **Full Validation Flow**:
    *   Train a dummy potential (or use a mock).
    *   Run validation on a small dataset.
    *   Check if `validation_report.json` is created.
