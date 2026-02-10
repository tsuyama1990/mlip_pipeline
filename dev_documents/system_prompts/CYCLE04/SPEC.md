# Cycle 04: Trainer & Active Learning

## 1. Summary

Cycle 04 focuses on the "Trainer" component, which is responsible for fitting the Atomic Cluster Expansion (ACE) potential to the accumulated DFT data. We will wrap the `Pacemaker` command-line tools (`pace_train`, `pace_activeset`, `pace_collect`) to automate the training process.

A critical feature in this cycle is "Active Set Selection" (D-Optimality). As the dataset grows, training on all structures becomes inefficient and prone to overfitting. We will implement logic to select a subset of "most informative" structures (Active Set) that maximize the determinant of the information matrix. This ensures that the potential learns efficiently from the most relevant data.

We will also implement "Delta Learning," where the trainer fits the residual error between the DFT energy and a physics-based baseline (LJ/ZBL). This is crucial for physical robustness.

## 2. System Architecture

The following file structure will be modified/created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── config.py         # Update TrainerConfig
│       ├── components/
│       │   ├── base.py
│       │   └── **trainer.py**    # Pacemaker Wrapper
│       └── utils/
│           └── **pacemaker_driver.py** # CLI Driver
└── tests/
    ├── **test_trainer.py**
    └── **test_pacemaker_driver.py**
```

## 3. Design Architecture

### Pacemaker Trainer (`components/trainer.py`)
The `PacemakerTrainer` implements the `BaseTrainer` interface. It manages the training lifecycle:
1.  **Update Dataset**: Add new DFT results to the persistent dataset (`accumulated.pckl.gzip`).
2.  **Select Active Set**: Use `pace_activeset` to filter the dataset.
3.  **Train**: Run `pace_train` with appropriate parameters (ladder schemes, cutoff, etc.).
4.  **Export**: Return the path to the trained `.yace` file.

### Active Set Logic
The `_select_active_set(dataset_path)` method:
*   Calls `pace_activeset` with `--max_size` (e.g., 500 structures) and `--criterion=d_opt`.
*   Reads the output (`active_set.pckl.gzip`).
*   Returns the path to the active set file.

### Delta Learning Setup
The `TrainerConfig` will include a `physics_baseline` field (e.g., `lj_cut`). The trainer must:
1.  Generate a `input.yaml` for Pacemaker that defines the baseline potential.
2.  Ensure `pace_train` respects this baseline.

### Pacemaker Driver (`utils/pacemaker_driver.py`)
A utility module to construct and execute shell commands for Pacemaker tools.
*   `run_pace_collect(atoms_list, output_path)`
*   `run_pace_activeset(input_path, output_path, max_size)`
*   `run_pace_train(dataset_path, initial_potential, output_dir)`

## 4. Implementation Approach

1.  **Driver Implementation**: Implement `utils/pacemaker_driver.py`. Use `subprocess.run` with proper error handling and logging.
2.  **Trainer Implementation**: Implement `components/trainer.py`. Wire it to the driver.
3.  **Dataset Management**: Implement logic to append new data to existing pickle files. Use `ase.io.read/write` or `pickle` directly if format allows.
4.  **Configuration**: Update `TrainerConfig` to include `active_set_size`, `max_epochs`, and `baseline_potential`.
5.  **Integration**: Update the `Orchestrator` to use the real `PacemakerTrainer`.

## 5. Test Strategy

### Unit Testing
*   **Command Construction**: Verify that `PacemakerTrainer` constructs the correct CLI arguments for `pace_train` (e.g., `--dataset data.pckl --output-dir out`).
*   **Active Set Mock**: Create a mock `run_pace_activeset` that simply copies the input file to the output. Assert that the trainer uses this output for training.

### Integration Testing (with Mocks)
*   **Full Training Cycle**:
    *   Create a small dataset of 10 structures (ASE Atoms).
    *   Call `trainer.update_dataset(structures)`.
    *   Call `trainer.train()`.
    *   Assert that `pace_train` was called (mocked subprocess).
    *   Assert that a `.yace` file path is returned.

### Integration Testing (Real Pacemaker - Optional)
*   **Small Fit**: If Pacemaker is installed, run a real training on 5 atoms.
*   **Verification**: Check if the resulting `.yace` file can be loaded by `pyace` or `lammps`.
