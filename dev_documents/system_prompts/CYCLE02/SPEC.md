# CYCLE 02 Specification: Trainer & Baseline Integration

## 1. Summary

In Cycle 02, we replace the `MockTrainer` with a functional `PacemakerTrainer` that interfaces with the actual `pace_train` binary. The critical feature here is "Delta Learning". We must ensure that the system can automatically configure a physical baseline (Lennard-Jones or ZBL) and train the ACE potential to learn only the residual energy. This guarantees that the final potential has physically correct core repulsion, preventing MD crashes. We will also implement `DatasetManager` to handle the serialization of `ase.Atoms` into Pacemaker's expected `.pckl.gzip` format.

## 2. System Architecture

New files and modifications to the file tree:

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── dataset.py           # [CREATE] Dataset abstractions
├── services/
│   ├── external/
│   │   ├── pacemaker_interface.py  # [CREATE] Wrapper for pace_train/activeset
│   │   └── lammps_tools.py         # [CREATE] Utils for LJ/ZBL parameters
│   └── trainer.py           # [CREATE] Concrete PacemakerTrainer
└── utils/
    └── file_io.py           # [CREATE] Helper for .pckl.gzip
```

## 3. Design Architecture

### Dataset Models (`dataset.py`)
-   `TrainingSet`: Abstraction for a collection of labeled `ase.Atoms`.
-   **Responsibility**: Validating that all atoms have "energy" and "forces" arrays before saving.

### Pacemaker Interface (`pacemaker_interface.py`)
-   Wraps the CLI calls to `pace_train`, `pace_activeset`.
-   **Input**: `TrainingConfig` model.
-   **Output**: Path to the generated `.yace` file.

### Baseline Logic (`lammps_tools.py`)
-   **Function**: `generate_baseline_config(elements)`.
-   **Logic**:
    -   Lookup atomic radii/weights.
    -   Generate a `potential.yaml` (Pacemaker format) that specifies `pair_style hybrid/overlay pace zbl`.
    -   This file tells Pacemaker to subtract the ZBL energy from the DFT energy *before* fitting.

## 4. Implementation Approach

1.  **Implement `DatasetManager`**:
    -   Use `pickle` and `gzip` to read/write the list of `ase.Atoms`.
    -   Ensure compatibility with Pacemaker's expected format (list of `ase.Atoms`).

2.  **Implement `PacemakerTrainer`**:
    -   Write the `train()` method.
    -   Step A: Convert the input list of atoms to `train.pckl.gzip`.
    -   Step B: Generate `input.yaml` for Pacemaker, injecting the correct ZBL/LJ baseline settings.
    -   Step C: Run `subprocess.run(["pace_train", ...])`.
    -   Step D: Return the path to the output `.yace` file.

3.  **Update Orchestrator**:
    -   Replace `MockTrainer` with `PacemakerTrainer` (can still be toggled via config).

## 5. Test Strategy

### Unit Testing
-   **Baseline Generation**: Verify that `generate_baseline_config(["Fe", "Pt"])` produces a correct YAML string with ZBL parameters.
-   **Dataset Serialization**: Create a list of `ase.Atoms`, save it, load it back, and assert equality.

### Integration Testing
-   **Mocked Binary Call**: We don't want to run actual `pace_train` in CI if it's not installed.
    -   Use `unittest.mock.patch("subprocess.run")`.
    -   Verify that the correct command-line arguments (including `--initial_potential` and baseline flags) are passed to the subprocess.
    -   Mock the creation of the output file so the orchestrator thinks training succeeded.
