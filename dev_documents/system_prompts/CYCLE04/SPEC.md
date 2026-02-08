# Cycle 04 Specification: Trainer (Pacemaker Integration)

## 1. Summary

Cycle 04 focuses on the core machine learning engine: **Pacemaker**. We implement the `Trainer` component to interface with the `pace_train` and `pace_activeset` command-line tools. Key features include **Active Set Optimization**, which intelligently selects a subset of the most informative structures from the accumulated dataset to keep training costs manageable, and **Delta Learning**, which configures Pacemaker to learn the difference between the DFT energy and a physics-based baseline (LJ/ZBL), ensuring robustness.

## 2. System Architecture

Files in **bold** are the focus of this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── **pacemaker.py**    # Wrapper for pace_train
│   │   └── **activeset.py**    # Wrapper for pace_activeset
├── domain_models/
│   ├── **config.py**           # Add PacemakerConfig
```

## 3. Design Architecture

### 3.1. PaceTrainer (`components/trainer/pacemaker.py`)
Inherits from `BaseTrainer`.
-   **Config**: `cutoff` (float), `order` (int), `reference_potential` (str: "lj", "zbl", "none").
-   **Method `train(dataset, initial_potential)`**:
    1.  **Data Preparation**: Converts `Dataset` (JSONL) to Pacemaker's expected pickle format (`collect.pckl.gzip`).
    2.  **Active Set Selection**: Calls `activeset.select(dataset)` to filter structures if the dataset is too large (>10k).
    3.  **Config Generation**: Writes `input.yaml` for Pacemaker.
        -   If `reference_potential="zbl"`, adds the ZBL definition to the YAML.
    4.  **Execution**: Runs `pace_train input.yaml`.
    5.  **Output**: Captures the resulting `output_potential.yace` and wraps it in a `Potential` object.

### 3.2. Active Set Logic (`components/trainer/activeset.py`)
-   **Purpose**: Maximize information (D-optimality) while minimizing dataset size.
-   **Command**: `pace_activeset --dataset full_data.pckl --max_size 1000 ...`
-   **Integration**: The trainer uses this to create a `train_set.pckl` from the raw `dataset`.

## 4. Implementation Approach

1.  **Update Config**: Add `PacemakerConfig` to `domain_models/config.py`.
2.  **Data Converter**: Implement a helper to convert `list[Structure]` (ASE Atoms) to the Pandas DataFrame format used by Pacemaker (saved as `.pckl.gzip`).
3.  **Implement PaceTrainer**:
    -   Use `subprocess.run` to execute `pace_train`.
    -   Ensure `input.yaml` is dynamically generated based on `GlobalConfig`.
4.  **Mocking**: Since `pacemaker` might not be installed in the dev environment, create a "MockPacemaker" that just copies the initial potential to output (or creates a dummy file).

## 5. Test Strategy

### 5.1. Unit Testing
-   **Config Generation**: Verify that `PaceTrainer` generates a valid `input.yaml` with the correct cutoff and ZBL settings.
-   **Data Conversion**: Verify that `Dataset` -> `pckl` conversion works (using `pandas` and `ase`).

### 5.2. Integration Testing
-   **Scenario**: "Train Loop"
-   **Config**: `trainer: pacemaker`, `oracle: mock` (providing 10 structures).
-   **Mocking**: Mock the `subprocess.run` call to `pace_train`.
-   **Check**: Verify that the system attempts to run the command and correctly identifies the output `potential.yace`.
