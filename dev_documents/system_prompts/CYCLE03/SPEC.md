# Cycle 03 Specification: Trainer (Pacemaker Integration)

## 1. Summary

Cycle 03 focuses on the "Trainer" module, which interfaces with the `pacemaker` library to constructing the Atomic Cluster Expansion (ACE) potential. This module is responsible for converting DFT data into a format suitable for training, managing the training process (including hyperparameter control), and selecting the most informative structures using "Active Set" optimization.

A key feature to be implemented here is "Delta Learning". To ensure physical robustness, we enforce a physics-based baseline (Lennard-Jones or ZBL). The Trainer must coordinate the calculation of this baseline energy/force for every structure in the dataset, subtract it from the DFT labels, and train the ACE model on the residual. This ensures that the final potential (Baseline + ACE) behaves physically even in data-sparse regions.

## 2. System Architecture

**Files to be created/modified in this cycle are marked in Bold.**

```ascii
src/mlip_autopipec/
├── training/
│   ├── **__init__.py**
│   ├── **wrapper.py**          # PacemakerWrapper class
│   ├── **dataset.py**          # Dataset manipulation logic
│   └── **baseline.py**         # Delta Learning (LJ/ZBL) calculator
```

## 3. Design Architecture

### `PacemakerWrapper` (in `training/wrapper.py`)
*   **Responsibility**: subprocess wrapper for `pace_train`, `pace_activeset`, etc.
*   **Methods**:
    *   `train(dataset_path, initial_potential=None) -> potential_path`: Runs training. If `initial_potential` is provided, it performs fine-tuning (fewer epochs, lower learning rate).
    *   `select_active_set(candidate_path, current_potential) -> selected_path`: Runs `pace_activeset` to filter redundant data.

### `DatasetManager` (in `training/dataset.py`)
*   **Responsibility**: Handle the conversion between ASE `Atoms` objects and Pacemaker's binary format (`.pckl.gzip`).
*   **Logic**:
    *   Pacemaker requires data to be in a specific pickle format.
    *   We need to ensure that `energy`, `forces`, and `stress` are correctly mapped from the ASE atoms.

### `BaselineCalculator` (in `training/baseline.py`)
*   **Responsibility**: Compute $E_{base}$, $F_{base}$, $S_{base}$ for Delta Learning.
*   **Logic**:
    *   Accepts an `Atoms` object and a baseline config (e.g., LJ params).
    *   Returns the baseline values.
    *   The `DatasetManager` will use this to modify the labels before saving the training set: $E_{target} = E_{DFT} - E_{base}$.

## 4. Implementation Approach

1.  **Baseline Logic**: Implement `BaselineCalculator` using ASE's built-in LJ or ZBL calculators.
2.  **Dataset Conversion**: Implement the `ase_to_pacemaker` conversion function. This might require inspecting the `pacemaker` source code or documentation for the exact pickle structure (or using `pace_collect` via subprocess if a direct API is unavailable).
3.  **Wrapper Implementation**: Implement `PacemakerWrapper` using `subprocess.run`. Ensure proper logging of the training progress (streaming stdout).
4.  **Integration**: Update `Orchestrator` to initialize `PacemakerWrapper`.

## 5. Test Strategy

### Unit Testing
*   **Baseline**:
    *   Create two atoms at distance $r$. Calculate LJ energy manually and compare with `BaselineCalculator`.
*   **Dataset**:
    *   Create a list of ASE atoms with random forces.
    *   Convert to Pacemaker format.
    *   Load it back (if possible) or verify the file is created.

### Integration Testing
*   **Mock Training**:
    *   Mock `subprocess.run` for `pace_train`.
    *   Verify that the correct command-line arguments are passed (especially `--initial_potential` for fine-tuning).
    *   Simulate the creation of an output `.yace` file.
