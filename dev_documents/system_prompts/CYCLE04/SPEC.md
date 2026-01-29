# Cycle 04: The Learner (Pacemaker Integration)

## 1. Summary

Cycles 02 and 03 gave us the ability to generate data (MD) and label it (DFT). Cycle 04 closes the loop by implementing the **Trainer**. We use `pacemaker`, a powerful library for fitting Atomic Cluster Expansion (ACE) potentials.

The training phase is not merely a regression task. In a physics-informed pipeline, it involves several critical steps:
1.  **Dataset Management**: Converting raw ASE atoms (with DFT forces) into the proprietary `.pckl.gzip` format used by `pacemaker`.
2.  **Reference Potential Subtraction (Delta Learning)**: We do not learn the total energy directly. We learn $E_{ACE} = E_{DFT} - E_{Ref}$. This requires defining a reference potential (Lennard-Jones or ZBL) that captures the "trivial" physics (e.g., core repulsion), leaving the ACE model to focus on the complex many-body interactions. This significantly improves stability.
3.  **Active Set Selection**: As we accumulate thousands of structures, training on all of them becomes inefficient. We implement **D-Optimality** selection (`pace_activeset`) to choose a sparse subset of structures that maximize the information content (determinant of the feature matrix).
4.  **Fitting**: Executing `pace_train` with appropriate regularisation.

By the end of this cycle, the system will be able to take a list of `Structure` objects and output a `potential.yace` file that is ready for simulation.

## 2. System Architecture

We introduce the `physics/training` package.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **potential.py**        # Potential metadata and paths
│       │   └── config.py               # Update with TrainingConfig
│       ├── physics/
│       │   ├── training/
│       │   │   ├── **__init__.py**
│       │   │   ├── **pacemaker.py**    # Wrapper for pace_train/activeset
│       │   │   └── **dataset.py**      # Conversion ASE <-> .pckl.gzip
│       └── utils.py
└── tests/
    └── physics/
        └── training/
            └── **test_dataset.py**
```

### Component Interaction

1.  **Orchestrator** gathers labelled `Structure` objects from the Oracle.
2.  **`DatasetManager`** converts them to a Pandas DataFrame format suitable for `pacemaker` (using `pace_neutral` internal tools or custom conversion) and saves as `train.pckl.gzip`.
3.  **`PacemakerRunner`** executes `pace_activeset` (optional) to prune the dataset.
4.  **`PacemakerRunner`** constructs the `input.yaml` for training.
    -   Defines the ACE basis set (cutoff, body-order).
    -   Defines the reference potential (e.g., `ZBL` for short range).
5.  **`PacemakerRunner`** executes `pace_train`.
6.  **Output**: A `potential.yace` file is generated.

## 3. Design Architecture

### 3.1. Potential Domain Model (`domain_models/potential.py`)

-   **Class `Potential`**:
    -   `path`: `Path` (Path to .yace file).
    -   `format`: `Literal["ace", "mace", "gap"]` (For future proofing).
    -   `elements`: `List[str]`.
    -   `creation_date`: `datetime`.
    -   `metadata`: `Dict` (e.g., RMSE on training set).

### 3.2. Dataset Conversion (`physics/training/dataset.py`)
Pacemaker uses a specific pickle format containing pandas DataFrames. We need to bridge ASE Atoms to this format.
-   **Function `atoms_to_dataframe(atoms_list: List[Structure])`**:
    -   Must serialize `energy` (scalar), `forces` (Nx3), and `stress` (3x3).
    -   Must handle "isolated atoms" (reference energies) if required.

### 3.3. Training Configuration (`config.py`)
-   **Class `TrainingConfig`**:
    -   `batch_size`: `int`.
    -   `max_epochs`: `int`.
    -   `ladder_step`: `List[int]` (Basis set size steps).
    -   `kappa`: `float` (Weighting of forces vs energy).

## 4. Implementation Approach

### Step 1: Config & Models
-   Update `Config` to include `TrainingConfig`.
-   Create `domain_models/potential.py`.

### Step 2: Dataset Manager
-   Implement `physics/training/dataset.py`.
-   Since `pacemaker` is a library, try to import `pacemaker.data` if available. If not (to keep dependency loose), implement a subprocess call to `pace_collect` which is the standard CLI tool for data conversion.
-   **Recommendation**: Use `subprocess.run(["pace_collect", ...])` to convert extended XYZ files to `.pckl.gzip`. It's more stable than relying on internal APIs that change.

### Step 3: Pacemaker Runner
-   Implement `physics/training/pacemaker.py`.
-   **Method `train()`**:
    -   Generate `input.yaml` using `jinja2` template.
    -   Run `pace_train`.
    -   Parse the output log to extract final RMSE.
-   **Method `select_active_set()`**:
    -   Run `pace_activeset`.
    -   Return the path to the pruned dataset.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Dataset Conversion**:
    -   Create a list of ASE atoms. Write to extxyz.
    -   Mock `pace_collect` execution.
    -   Check that the input arguments to `pace_collect` are correct.

### 5.2. Integration Testing (Mocked)
-   **Mock Training**:
    -   We cannot run actual training in CI (too slow, requires tensor libraries).
    -   Mock `subprocess.run` for `pace_train`.
    -   Have the mock create a dummy `potential.yace` file and a `log.txt` with fake RMSE values.
    -   Assert that `PacemakerRunner` parses the fake RMSE correctly and returns a `Potential` object pointing to the dummy file.

### 5.3. Active Set Test
-   Mock `pace_activeset` to take a dataset of 100 structures and output a dataset of 10 structures.
-   Verify the Orchestrator flow uses this reduced dataset.
