# Cycle 03 Specification: Trainer Module (Pacemaker Integration)

## 1. Summary
This cycle implements the **Trainer** component, which interfaces with the **Pacemaker** suite to fit Atomic Cluster Expansion (ACE) potentials. The Trainer is responsible for managing the training dataset (accumulating labeled structures over cycles), configuring the fitting parameters (including physics-based baselines like ZBL or Lennard-Jones for Delta Learning), and executing the `pace_train` command. It also handles the "Active Set" selection (`pace_activeset`), which is crucial for data efficiency by selecting only the most informative structures for the basis set.

## 2. System Architecture

### 2.1. File Structure

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**     # [MODIFY] Add TrainerConfig
│       ├── domain_models/
│       │   └── **structure.py**        # [MODIFY] Add Dataset class logic
│       ├── infrastructure/
│       │   ├── **pacemaker/**
│       │   │   ├── **__init__.py**
│       │   │   ├── **adapter.py**      # [NEW] PacemakerTrainer implementation
│       │   │   └── **dataset.py**      # [NEW] Dataset conversion logic
└── tests/
    └── unit/
        └── **test_trainer.py**         # [NEW] Tests for PacemakerTrainer
```

## 3. Design Architecture

### 3.1. `TrainerConfig` (Pydantic)
*   `ace_basis_config`: Dict (radial/angular cutoffs, orders).
*   `physics_baseline`: str (e.g., `zbl`, `lj`, or `none`).
*   `fitting_params`: Dict (ladder_step, weighting schemes).
*   `active_set_size`: int (Target number of basis configurations).

### 3.2. `Dataset` Management
Pacemaker uses a specific `pickle.gzip` format for datasets.
*   **Internal Representation**: We maintain a list of `StructureMetadata` in Python memory (or a lightweight DB).
*   **Conversion**: Before training, we must serialize this list into the `pacemaker.data.Structure` format and save it to disk.
*   **Active Set**: We wrapping `pace_activeset` to filter this dataset.

### 3.3. `PacemakerTrainer` Class
Implements `BaseTrainer`.
*   **Responsibilities**:
    1.  `update_dataset(new_data)`: Add new structures to the persistent dataset.
    2.  `prepare_input()`: Generate `input.yaml` for Pacemaker.
    3.  `train()`: Run `pace_train` via subprocess.
    4.  `post_process()`: Check if `output_potential.yace` was created and move it to the `potentials/` directory.

### 3.4. Delta Learning Logic
To enforce physical robustness, we configure Pacemaker to fit the *residual* energy:
$E_{ACE} = E_{DFT} - E_{Baseline}$
The `PacemakerTrainer` must automatically configure the `variation` section in `input.yaml` to include `zbl` or `lj` parameters if specified in the config.

## 4. Implementation Approach

1.  **Dataset Logic**: Implement `infrastructure/pacemaker/dataset.py`.
    *   Since we cannot import `pyace` (it might not be installed in the dev environment), we might need to rely on the `pace_collect` CLI tool to merge `.xyz` or `.extxyz` files.
    *   **Strategy**: Write `new_data` to a temporary `data.extxyz`, then call `pace_collect`.
2.  **Implement Adapter**: Create `infrastructure/pacemaker/adapter.py`.
    *   Construct the `input.yaml` dictionary.
    *   Handle the `physics_baseline` logic (injecting ZBL parameters).
    *   Use `subprocess.run(["pace_train", ...])`.
    *   Stream stdout to the main logger.
3.  **Active Set Selection**:
    *   Implement a method `select_active_set()` that runs `pace_activeset` to identify key structures.
4.  **Integration**: Update `main.py` to use `PacemakerTrainer`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Config Generation**: Verify that `prepare_input()` produces valid YAML that matches the `TrainerConfig`.
*   **Command Construction**: Verify the `subprocess` calls.
    *   Did it point to the correct dataset path?
    *   Did it include the `--initial_potential` flag if fine-tuning?

### 5.2. Integration Testing (Mocked Binary)
*   **Mocking `pace_train`**: Create a dummy script or use `unittest.mock` to simulate the execution of `pace_train`.
*   **Behavior**:
    *   The mock should read `input.yaml`.
    *   It should create a dummy `output_potential.yace` file.
    *   It should exit with code 0.
*   **Test**: Run `trainer.train()`. Assert that it returns the path to the new yace file.
