# Cycle 04 Specification: The Trainer (Pacemaker Integration)

## 1. Summary

Cycle 04 integrates the core machine learning engine: **Pacemaker**. The Trainer module is responsible for taking the labeled `Dataset` from the Oracle and fitting an Atomic Cluster Expansion (ACE) potential.

This cycle implements two critical features for efficiency and robustness:
1.  **Delta Learning**: The potential is trained as a correction to a physical baseline (ZBL/LJ), ensuring stability in the repulsive regime.
2.  **Active Set Selection**: Instead of training on every single structure, we use D-optimality to select the most informative subset, keeping the regression problem tractable.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── config/
│   └── config_model.py         # Update TrainerConfig
├── infrastructure/
│   └── trainer/
│       ├── **__init__.py**
│       └── **pacemaker_trainer.py** # Wrapper for pace_train/pace_activeset
└── domain_models/
    └── **potential.py**        # Model for potential artifacts
```

## 3. Design Architecture

### 3.1. Trainer Configuration
We extend `TrainerConfig`:
*   `type`: Literal["mock", "pacemaker"]
*   `cutoff`: float (e.g., 5.0)
*   `b_ref`: Dict[str, Any] (Baseline config, e.g., `{"type": "zbl", "inner_cutoff": 1.0}`)
*   `ace_params`: Dict[str, Any] (Basis set size, etc.)

### 3.2. PacemakerTrainer Logic
*   **Inheritance**: Inherits from `BaseTrainer`.
*   **Method**: `train(...)`.
    1.  **Convert**: Transform ASE `Dataset` -> Pacemaker `.pckl.gzip`.
    2.  **Filter**: Run `pace_activeset` to select structures (optional, based on config).
    3.  **Config**: Generate `input.yaml` for Pacemaker, injecting the `b_ref` settings.
    4.  **Execute**: Call `pace_train` via subprocess.
    5.  **Artifact**: Return path to `output_potential.yace`.

## 4. Implementation Approach

1.  **File I/O**: Implement efficient conversion from ASE Atoms list to Pacemaker's expected pickle format.
2.  **Config Generation**: Write a Jinja2 template or Python builder for `potential_config.yaml`. It must support dynamic element lists (e.g., if system is Fe-Pt, generate mappings for Fe and Pt).
3.  **Subprocess Management**: Robust wrapper around `subprocess.run` to handle `pace_train` execution, capturing stdout/stderr for logging.
4.  **Delta Learning**: Ensure the `b_ref` section in the config correctly specifies ZBL parameters.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Gen**: Feed `TrainerConfig` with Fe-Pt system. Assert the generated YAML string contains `Fe` and `Pt` sections and the correct ZBL parameters.
*   **Conversion**: Create dummy atoms. specific properties (forces, energy) must be preserved in the pickle.

### 5.2. Integration Testing (Mocked Binary)
*   **Scenario**: `pace_train` is not installed.
*   **Mock**: Create a fake `pace_train` script (or mock subprocess) that simply copies a dummy `potential.yace` to the output.
*   **Assertion**: The `PacemakerTrainer` should run through the whole flow (data write -> config write -> process call -> result return) without error.
