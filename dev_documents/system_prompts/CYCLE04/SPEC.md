# Cycle 04 Specification: Pacemaker Learner

## 1. Summary

Cycle 04 implements the **Trainer** module, which interfaces with the `pacemaker` library to train the Atomic Cluster Expansion (ACE) potential. This module handles the conversion of DFT results into the specific dataset format required by Pacemaker (`.pckl.gzip`), configures the training parameters (including Delta Learning with a ZBL/LJ baseline), and executes the training process. It also integrates "Active Set" selection to filter redundant data and optimize training efficiency.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── config.py                     # Update: Add TrainingConfig
│   └── **potential.py**              # Potential model
├── modules/
│   └── **trainer/**
│       ├── **__init__.py**
│       ├── **pacemaker.py**          # Pacemaker wrapper
│       └── **dataset.py**            # Dataset management
└── orchestration/
    └── phases/
        ├── **__init__.py**
        └── **training.py**           # TrainingPhase implementation
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`TrainingConfig`**:
    *   `ace_basis_config`: Dict (B-basis settings: N, L, r_cut)
    *   `reference_potential`: Enum (None, ZBL, LJ)
    *   `max_epochs`: int
    *   `batch_size`: int
    *   `loss_weights`: Dict (energy, force, stress)

#### `potential.py`
*   **`Potential`**:
    *   `path`: Path to `.yace` file.
    *   `meta`: Training history, loss values.

### Components (`modules/trainer/`)

#### `dataset.py`
*   **`DatasetBuilder`**:
    *   **`add_data(dft_results)`**: Appends new data to the accumulated set.
    *   **`export_for_pacemaker(path)`**: Writes data to `train.pckl.gzip`.
    *   **`select_active_set(method='maxvol')`**: Filters the dataset using `pace_activeset`.

#### `pacemaker.py`
*   **`PacemakerRunner`**:
    *   Wraps `pace_train` command via `subprocess`.
    *   **`generate_input_yaml(config)`**: Creates `input.yaml` for Pacemaker.
    *   **`train(dataset_path, output_dir)`**: Runs the training.
    *   **`configure_reference(ref_type)`**: Sets up ZBL/LJ reference logic.

### Orchestration (`orchestration/phases/training.py`)

#### `TrainingPhase`
*   Compiles all successful `DFTResult`s from previous cycles.
*   Updates the master dataset.
*   Runs active set selection (if enabled).
*   Executes `PacemakerRunner.train`.
*   Updates `WorkflowState` with the path to the new potential.

## 4. Implementation Approach

1.  **Update Config**: Add `TrainingConfig`.
2.  **Implement Dataset Builder**:
    *   Use `ase.io.write` to save in formats Pacemaker can read (extxyz or pckl if possible).
    *   Implement Active Set selection logic (calling `pace_activeset`).
3.  **Implement Runner**:
    *   Create logic to generate valid Pacemaker YAML configs.
    *   Ensure Delta Learning (reference potential) is correctly configured in the YAML.
4.  **Implement Training Phase**:
    *   Connect the data flow from Oracle to Trainer.
    *   Ensure output potentials are versioned (e.g., `potential_gen_01.yace`).

## 5. Test Strategy

### Unit Testing
*   **`test_dataset.py`**:
    *   Create a dummy `DFTResult`.
    *   Verify it can be converted to the format expected by Pacemaker.
*   **`test_pacemaker_runner.py`**:
    *   Verify `input.yaml` generation matches the config.
    *   Check specifically for `reference_potential` settings.

### Integration Testing
*   **`test_training_phase.py`**:
    *   Mock the `pace_train` command (don't run actual heavy training).
    *   Verify that the phase produces a "potential file" (mocked) and updates the state.
