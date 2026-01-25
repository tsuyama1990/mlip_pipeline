# Cycle 03: Trainer (Pacemaker Interface)

## 1. Summary
This cycle integrates the **Pacemaker** engine, enabling the system to train Machine Learning Potentials (MLIPs). We implement the `Trainer` module, which handles dataset serialization (conversion of ASE structures to Pacemaker format), configuration of the learning process (Delta Learning with physical baselines), and execution of the training loop.

## 2. System Architecture

We add the `phases/training` module.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── orchestration/
│   └── phases/
│       └── **training/**
│           ├── **__init__.py**
│           ├── **manager.py**       # TrainingPhase implementation
│           ├── **dataset.py**       # Data conversion & management
│           ├── **pacemaker.py**     # Wrapper for pace_train/pace_collect
│           └── **priors.py**        # Reference potential (ZBL/LJ) config
└── tests/
    └── **test_training.py**
```

## 3. Design Architecture

### Dataset Management (`dataset.py`)
*   **Format**: Pacemaker uses gzip-compressed pickle files (`.pckl.gzip`) containing `pandas.DataFrame`.
*   **Responsibility**: Convert a list of `ASE Atoms` (from Oracle) into this specific format. It must strictly handle Energy, Force, and Stress columns.
*   **Accumulation**: Maintain a cumulative dataset (`accumulated.pckl.gzip`) that grows with each cycle.

### Pacemaker Wrapper (`pacemaker.py`)
*   **Execution**: Calls `pace_train` via `subprocess`.
*   **Configuration**: Generates the `input.yaml` for Pacemaker dynamically, injecting parameters from `WorkflowConfig` (cutoff, ladder, etc.).
*   **Initial Potential**: Supports fine-tuning by passing `--initial_potential` pointing to the previous cycle's `yace` file.

### Physical Priors (`priors.py`)
*   Generates the configuration block for reference potentials (ZBL or LJ). This is critical for the "Physics-Informed" aspect, ensuring short-range repulsion.

## 4. Implementation Approach

1.  **Dataset Conversion**: Implement `DatasetManager.save(atoms_list, path)`. Use `pyamtgen` or direct DataFrame construction if Pacemaker API is not available (Pacemaker is primarily CLI-driven).
2.  **Config Generation**: Implement a builder that takes `TrainingConfig` and writes `input.yaml` for Pacemaker.
3.  **Wrapper**: Implement `PacemakerWrapper.train()`. It should run the training in a temporary directory and move the resulting `potential.yace` to the `potentials/` directory.
4.  **Integration**: `TrainingPhase.run()` orchestrates the flow: Load Data -> Update Accumulation -> Generate Config -> Train -> Return path to new potential.

## 5. Test Strategy

### Unit Testing
*   **`test_dataset.py`**: Create dummy atoms and verify they can be saved and loaded back as a DataFrame with correct columns.
*   **`test_priors.py`**: Verify that ZBL parameters (charge, radius) are correctly generated for given elements.

### Integration Testing
*   **Mock Training**: Since training takes time, create a mock `pace_train` command that just copies a dummy `potential.yace` to the output. Verify the pipeline handles file paths correctly.
