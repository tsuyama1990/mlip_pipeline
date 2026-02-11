# Cycle 04 Specification: Trainer (Pacemaker & Active Set)

## 1. Summary
This cycle integrates the "Learner" component: the `Trainer`. This module interfaces with the `pacemaker` library to fit Atomic Cluster Expansion (ACE) potentials. The training process is not a simple "fit all data" operation; it involves intelligent data selection and physical regularization. The module implements **Active Set Selection** (D-Optimality) to filter redundant structures from the training set, keeping the regression problem tractable. It also enforces **Delta Learning**, ensuring that the ACE model only learns the difference between the DFT energy and a physics-based baseline (ZBL or LJ), which provides robustness in the extrapolation regime.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── base.py                 # [CREATE] Abstract Base Class
│   │   ├── pacemaker.py            # [CREATE] Pacemaker CLI Wrapper
│   │   ├── active_set.py           # [CREATE] MaxVol Selection Logic
│   │   └── dataset_utils.py        # [CREATE] .pckl.gzip management
├── domain_models/
│   ├── config.py                   # [MODIFY] Add TrainingConfig
│   └── enums.py                    # [MODIFY] Add TrainerType
└── core/
    └── orchestrator.py             # [MODIFY] Integrate Trainer into train()
```

### 2.2. Component Interaction
1.  **`Orchestrator`**:
    *   Collects new labeled structures from Oracle.
    *   Calls `trainer.update_dataset(new_data)`.
    *   Calls `trainer.train(context=state)`.
2.  **`PacemakerWrapper`**:
    *   **Data Merging**: Adds new structures to `train.pckl.gzip`.
    *   **Active Set Selection**: Runs `pace_activeset` to select top $N$ informative structures.
    *   **Training**: Runs `pace_train` with Delta Learning settings.
    *   **Output**: Produces `potential.yace`.
3.  **`Orchestrator`** updates `state.current_potential_path`.

## 3. Design Architecture

### 3.1. Domain Models

#### `config.py`
*   `TrainingConfig`:
    *   `ace_basis_size`: str (e.g., "huge", "medium")
    *   `max_epochs`: int
    *   `batch_size`: int
    *   `active_set_size`: int (e.g., 500)
    *   `physics_baseline`: Literal['ZBL', 'LJ', 'None']

### 3.2. Core Logic

#### `active_set.py`
*   **Responsibility**: Reduce data redundancy.
*   **Method**: `select_active_set(dataset_path, n_select)`.
*   **Implementation**: Wraps `pace_activeset` command. It computes the B-matrix (basis functions) and selects rows that maximize the determinant.

#### `pacemaker.py`
*   **Responsibility**: Execute training.
*   **Logic**:
    *   Constructs `input.yaml` for Pacemaker.
    *   Crucial: If `physics_baseline` is set, it must calculate the baseline energy for all atoms and subtract it *before* passing to ACE fitting? Actually, Pacemaker handles Delta Learning via `energy_calculator: type: python ...`. However, the simpler approach supported by Pacemaker is to define a reference potential in the YAML.
    *   *Correction*: We will use the standard Pacemaker approach of defining `pair_style hybrid/overlay` during MD, but for training, we might need to rely on Pacemaker's internal reference potential handling or pre-subtraction. For Cycle 04, we will assume pre-subtraction or standard fitting, to be refined. *Refinement*: The architecture specifies "Delta Learning". We will implement `ZBL` subtraction in `dataset_utils.py` if Pacemaker doesn't support it natively in the config, OR configure Pacemaker to use a reference potential. (Pacemaker supports `reference_potential` in input.yaml).

## 4. Implementation Approach

### Step 1: Interface Definition
*   Define `BaseTrainer` in `components/trainer/base.py`.
*   Define `train(...) -> Path`.

### Step 2: Dataset Management
*   Implement `dataset_utils.py`.
*   Function to convert List[Atoms] to Pacemaker's pandas/pickle format.
*   Function to merge two datasets.

### Step 3: Active Set Wrapper
*   Implement `ActiveSetSelector`.
*   Use `subprocess` to call `pace_activeset`.

### Step 4: Training Wrapper
*   Implement `PacemakerWrapper`.
*   Generate `input.yaml` dynamically.
*   Call `pace_train`.

### Step 5: Orchestrator Integration
*   Update `Orchestrator.train()` to manage the `potentials/` directory.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_dataset_utils.py`**:
    *   Create dummy Atoms. Save to pckl. Load back. Assert equality.
*   **`test_pacemaker_config.py`**:
    *   Verify `input.yaml` generation. Check if `max_num_epochs` matches config.

### 5.2. Integration Testing
*   **`test_training_mock.py`**:
    *   Since running real `pace_train` is slow/heavy, we might need a "Mock Trainer" that just touches a `potential.yace` file.
    *   *Real Pacemaker Test*: If `pacemaker` is installed in CI, run a tiny training (1 epoch, 10 structures). Assert `output_potential.yace` is created.
