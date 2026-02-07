# Cycle 04 Specification: Trainer & Pacemaker Interface

## 1. Summary
This cycle implements the "Trainer" component, which acts as the bridge to the Pacemaker library. Its primary responsibility is to take the labelled data (from the Oracle), convert it into the format required by Pacemaker (`.pckl.gzip`), and execute the training process to produce an ACE potential (`.yace`). Crucially, it also implements "Active Set" selection to filter the training data, ensuring that we only train on the most informative structures (D-optimality), thereby keeping the dataset compact and the training efficient.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the primary deliverables for this cycle.

```ascii
src/
└── mlip_autopipec/
    ├── implementations/
    │   └── **trainer/**
    │       ├── **__init__.py**
    │       ├── **pacemaker_trainer.py** # Main Class
    │       └── **data_manager.py**      # Data Conversion & Active Set
    └── utils/
        └── **runner.py**                # Subprocess Helper
```

## 3. Design Architecture

### 3.1. PacemakerTrainer
The `PacemakerTrainer` class implements the `BaseTrainer` interface.
-   **Configuration**: Takes `TrainerConfig` which specifies the ACE basis set parameters (cutoff, order, etc.) and fitting hyperparameters.
-   **Execution**: It constructs the `pace_train` command line arguments and executes it via `subprocess`. It parses the stdout to track progress (RMSE).

### 3.2. Data Manager & Active Set
The `DataManager` handles dataset operations.
-   **Conversion**: Converts a list of `Structure` objects (with energy/forces) into a pandas DataFrame and saves it as `dataset.pckl.gzip` (Pacemaker's native format).
-   **Active Set**: Wraps the `pace_activeset` command. It takes a large pool of candidate structures and selects a subset that maximises the information matrix determinant (D-optimality). This is key for the "Active Learning" efficiency.

## 4. Implementation Approach

### Step 1: Data Conversion
Implement `data_manager.py`.
-   Use `pandas` to create the dataframe. Ensure columns `ase_atoms`, `energy`, `forces` are correctly formatted.
-   Implement `save_dataset(structures, path)`.

### Step 2: Active Set Selection
Implement `select_active_set(candidates, current_dataset)` in `data_manager.py`.
-   Call `pace_activeset` command.
-   Return the indices/subset of selected structures.

### Step 3: Trainer Implementation
Implement `PacemakerTrainer` in `pacemaker_trainer.py`.
-   `train(dataset_path, initial_potential=None) -> potential_path`.
-   Generate `input.yaml` for Pacemaker.
-   Run `pace_train`.
-   Return the path to the best potential found.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Data Conversion**: Create dummy `Structure` objects. Convert to `.pckl.gzip`. Load it back (using pandas) and assert data integrity.
-   **Config Generation**: Assert that the generated `input.yaml` for Pacemaker contains the correct parameters from `TrainerConfig`.

### 5.2. Integration Testing (Mocked Pacemaker)
-   Since Pacemaker might not be installed in the CI environment, we can mock the `pace_train` command (similar to `mock_pw.py` in Cycle 03) or mock the `subprocess.run` call.
-   Verify that the `Trainer` correctly constructs the command string and handles the output file paths.
