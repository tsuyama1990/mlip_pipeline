# Cycle 04 Specification: The Trainer (Pacemaker Integration)

## 1. Summary

Cycle 04 introduces the learning capability (The "Learner"). Up to this point, we have generated structures (Explorer) and calculated their true energies (Oracle). Now, we implement the `PacemakerTrainer` to fit an ACE (Atomic Cluster Expansion) potential to this data. This cycle involves wrapping the external `pacemaker` CLI tools (`pace_train`, `pace_collect`, `pace_activeset`) into a Python interface. Key features include the "Hybrid Potential" setup (enforcing a physical baseline like ZBL) and "Active Set Selection" (using D-Optimality to select the most informative structures from a large pool), ensuring high data efficiency.

## 2. System Architecture

### 2.1 File Structure

**Bold** files are to be created or modified in this cycle.

```text
.
├── src/
│   └── mlip_autopipec/
│       ├── physics/
│       │   ├── trainer/
│       │   │   ├── **__init__.py**
│       │   │   ├── **pacemaker.py**    # Wraps pace_* commands
│       │   │   └── **dataset_utils.py**# Helpers for pckl.gzip
│       │   └── **__init__.py**
│       ├── validation/
│       │   ├── **__init__.py**
│       │   └── **basic_metrics.py**    # RMSE calculation
│       └── orchestration/
│           └── orchestrator.py         # Update to use real Trainer
└── tests/
    ├── unit/
    │   └── **test_pacemaker_wrapper.py**
    └── integration/
        └── **test_training_loop.py**
```

## 3. Design Architecture

### 3.1 `PacemakerTrainer` Class
-   **Role**: Manages the training lifecycle.
-   **Responsibilities**:
    -   **Data Ingestion**: Convert list of `ase.Atoms` (from Oracle) into Pacemaker's native `dataframe.pckl.gzip` format using `pace_collect` (or direct Pandas manipulation if feasible).
    -   **Active Set Selection**: Before training, run `pace_activeset` to filter redundant structures based on the MaxVol algorithm.
    -   **Training**: Execute `pace_train`. Needs to construct the input YAML file for Pacemaker dynamically (setting cutoffs, B-basis size, etc., from `TrainingConfig`).
    -   **Output Management**: Locate the resulting `.yace` file and store it in the `potentials/` directory.

### 3.2 Hybrid Potential Logic
-   **Concept**: $V_{total} = V_{ZBL} + V_{ACE}$.
-   **Implementation**:
    -   The Trainer doesn't just train ACE; it must configure the training to learn the *difference* between DFT and the Baseline.
    -   *Configuration*: The generated `input.yaml` for Pacemaker must specify the reference potential (e.g., `embeddings: { ... }`).

### 3.3 Basic Validation (`basic_metrics.py`)
-   **Role**: Immediate sanity check after training.
-   **Metrics**:
    -   RMSE (Root Mean Square Error) for Energy (meV/atom) and Forces (eV/A).
    -   If RMSE > Threshold (e.g., 50 meV/atom), the Trainer should flag a warning or raise a `TrainingDivergedError`.

## 4. Implementation Approach

1.  **Dataset Utilities**:
    -   Implement `atoms_to_pacemaker(atoms_list, filename)`: Writes the ASE atoms to the specific pickle format Pacemaker expects. (Check Pacemaker docs: usually it reads `extxyz` via `pace_collect`).
    -   *Decision*: Use `pace_collect` via subprocess for reliability.

2.  **Pacemaker Wrapper**:
    -   `train(dataset_path, initial_potential=None)`:
        1. Construct command: `pace_train ...`.
        2. Handle `stdout`/`stderr` logging.
        3. Parse the output logs to extract training loss curves (optional but good for visibility).
    -   `select_active_set(candidate_path)`:
        1. Run `pace_activeset`.
        2. Return the path to the filtered dataset.

3.  **Config Integration**:
    -   Map `TrainingConfig` fields (e.g., `l_max`, `rcut`) to the `pace_train` arguments or `input.yaml`.

4.  **Orchestrator Update**:
    -   In `run_cycle`, after Oracle returns data:
        1. Call `trainer.update_dataset(new_data)`.
        2. Call `trainer.train()`.
        3. Call `validator.check_metrics()`.

## 5. Test Strategy

### 5.1 Unit Testing
-   **`test_pacemaker_wrapper.py`**:
    -   **Command Construction**: Verify that `train()` builds the correct CLI string (e.g., proper flags for `--dataset`, `--output-dir`).
    -   **Mock Subprocess**: Use `unittest.mock.patch('subprocess.run')` to simulate `pace_train` success/failure without actually installing Pacemaker in the test environment (for unit tests).

### 5.2 Integration Testing
-   **`test_training_loop.py`**:
    -   **Requirement**: Needs `pacemaker` installed (or skipped if not).
    -   **Scenario**:
        1. Create a synthetic dataset (10 structures of perturbed Silicon).
        2. Run `PacemakerTrainer.train()`.
        3. Check if `output_potential.yace` is created.
        4. Load the potential with `ase.calculators.lammpsrun` (or `pyacemaker.calculator`) and compute forces on a test atom.
    -   **Validation**: Assert `forces` are not None.
