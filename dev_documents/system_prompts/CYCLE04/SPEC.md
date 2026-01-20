# Cycle 04: Active Learning & Training Loop

## 1. Summary

Cycle 04 marks the pivotal transition from "Data Collection" to "Knowledge Creation". In previous cycles, we established the infrastructure to generate structures (Cycle 2), filter them with a surrogate (Cycle 3), and calculate their ground-truth properties using DFT (Cycle 1). Now, we implement **Module D: Active Learning & Training**. This module is the "Brain" of the operation, responsible for digesting the accumulated DFT data and synthesizing it into a fast, accurate Machine Learning Interatomic Potential (MLIP).

We have selected **Pacemaker** as our primary training engine. Pacemaker fits **Atomic Cluster Expansion (ACE)** potentials. ACE potentials offer a unique value proposition: they are mathematically rigorous (based on body-ordered expansions of atomic density), computationally efficient (comparable to EAM potentials), and physically interpretable. Unlike deep Graph Neural Networks (GNNs) which can be black boxes, ACE potentials are robust enough for the massive, long-timescale MD simulations we plan in Cycle 5.

The core innovation in this cycle is the automated **Delta Learning** workflow. Learning the absolute Total Energy of a system is difficult because the energy landscape is dominated by the steep Pauli repulsion at short interatomic distances. If the ML model fits this region poorly, atoms can fuse together during simulation. To solve this, we implement a **Physics-Informed Baseline**:
1.  **ZBL Potential**: We explicitly calculate the Ziegler-Biersack-Littmark (ZBL) repulsion energy for every atom pair in the dataset. This captures the $1/r$ Coulombic behavior and screening effects.
2.  **Delta Target**: The ML model is trained only on the *residual* (difference): $E_{ML} = E_{DFT} - E_{ZBL}$.
This approach ensures that the final potential $V_{total} = V_{ACE} + V_{ZBL}$ always has the correct repulsive physics, preventing catastrophic failures in high-temperature simulations (e.g., melt-quench).

By the end of this cycle, the system will be able to autonomously export data from the ASE database, configure a Pacemaker training job with dynamic hyperparameters (loss weights, cutoffs, basis sizes), execute the training, and validate the resulting `.yace` potential against a holdout test set.

## 2. System Architecture

### 2.1. Code Blueprint and File Structure

The training module is designed to abstract the details of the specific MLIP code (Pacemaker), allowing future extensions to other codes (e.g., NequIP/Allegro).

The following file structure will be implemented. Files in **bold** are the primary deliverables.

```
mlip_autopipec/
├── training/
│   ├── **__init__.py**
│   ├── **config.py**               # Pydantic schemas for Training parameters (Cutoffs, Loss weights)
│   ├── **models.py**               # Data structures for Training Batches
│   ├── **physics.py**              # Implementation of the ZBL/Baseline potentials
│   ├── **dataset_prep.py**         # Logic for ZBL subtraction and file formatting (.p4p/.xyz)
│   ├── **pacemaker_wrapper.py**    # The orchestrator for the external training process
│   └── **templates/**              # Jinja2 templates for input files
│       └── **input.yaml.j2**       # Pacemaker configuration template
└── tests/
    └── training/
        ├── **test_physics.py**
        ├── **test_dataset_prep.py**
        └── **test_pacemaker.py**
```

### 2.2. Component Interaction and Data Flow

1.  **Trigger**:
    The Orchestrator initiates the cycle by calling `PacemakerWrapper.train(generation_id)`.

2.  **Data Retrieval**:
    The wrapper queries the `DatabaseManager` for all structures tagged with `status=TRAINING_READY`.
    -   It implements a **Train/Test Split**: 90% of data is used for training, 10% is held out for validation. The split is stratified by `config_type` to ensure the test set covers all phases (liquid, solid, distorted).

3.  **Preprocessing (The "Secret Sauce")**:
    The `DatasetBuilder` iterates through every structure.
    -   **Physics Baseline**: It calls `physics.ZBLCalculator` to compute $E_{ZBL}$ and $F_{ZBL}$.
    -   **Delta Calculation**: It subtracts the baseline: $E_{target} = E_{DFT} - E_{ZBL}$, $F_{target} = F_{DFT} - F_{ZBL}$.
    -   **Force Masking**: It checks for `force_mask` arrays (created in Cycle 5). If present, these are written as weights (0.0 or 1.0).
    -   **Formatting**: The data is written to disk in the **Extended XYZ** format required by Pacemaker (`Properties=species:S:1:pos:R:3:forces:R:3:energy:R:1...`).

4.  **Configuration Generation**:
    The wrapper loads the `TrainConfig`. It uses `jinja2` to render the `input.yaml` file.
    -   It sets the `cutoff` (e.g., 5.0 Angstrom).
    -   It sets the `loss_weights` (typically Forces have 10x-100x higher weight than Energy).
    -   It defines the Basis Set (B-basis size, Max Degree).

5.  **Execution**:
    The `pacemaker` binary is launched as a subprocess. The wrapper monitors `stdout` for progress bars and logs them to the central logger.

6.  **Validation and Artifacts**:
    -   Upon completion, the wrapper parses the `metrics.json` output to retrieve the final RMSE values.
    -   The resulting `potential.yace` file is moved to the `output/potentials/` directory.
    -   The training metadata (RMSE, epochs, time) is saved to the DB.

## 3. Design Architecture

### 3.1. Training Configuration (`training/config.py`)

-   **`TrainConfig`**:
    -   `cutoff`: `float` (default 5.0). The interaction radius.
    -   `b_basis_size`: `int` (default 500). Number of radial basis functions.
    -   `max_degree`: `int` (default 3). Angular correlation order (Body order - 1).
    -   `energy_weight`: `float` (default 1.0).
    -   `force_weight`: `float` (default 100.0). Forces provide 3N data points vs 1 Energy, so they dominate the loss.
    -   `kappa`: `float`. Regularization parameter.

### 3.2. Physics Module (`training/physics.py`)

This module provides the analytical baselines. It must be extremely efficient as it runs on every atom in the dataset.

-   **Class**: `ZBLCalculator`
-   **Method**: `get_potential_energy(atoms) -> float`
    -   **Math**: Sum over pairs $V_{ZBL}(r_{ij}) = \frac{Z_i Z_j e^2}{r_{ij}} \Phi(\frac{r_{ij}}{a})$.
    -   **Implementation**: Vectorized NumPy calculation using `scipy.spatial.distance.pdist`.
    -   **Optimization**: Pre-compute the screening function $\Phi$ coefficients.
-   **Method**: `get_forces(atoms) -> Array[Nx3]`
    -   **Math**: Analytical gradient $-\nabla V_{ZBL}$.

### 3.3. Dataset Builder (`training/dataset_prep.py`)

Handles the complex I/O requirements of the external trainer.

-   **Class**: `DatasetBuilder`
-   **Method**: `prepare_dataset(structures: List[Atoms], output_path: Path)`
    -   **Logic**:
        1.  Open output file.
        2.  For each atom `s` in structures:
            -   Calc baseline $E_b, F_b$.
            -   Update `s.info['energy'] -= E_b`.
            -   Update `s.arrays['forces'] -= F_b`.
            -   Write to file with correct headers.
    -   **Safety**: Explicitly checks for `NaN` or `Inf` values. If found, skips the structure and logs a warning.

### 3.4. Pacemaker Wrapper (`training/pacemaker_wrapper.py`)

Orchestrates the external tool.

-   **Class**: `PacemakerWrapper`
-   **Method**: `run_training(dataset_path, test_path, config) -> TrainingResult`
    -   **Step 1 (Setup)**: Create a temporary run directory. Write `input.yaml`.
    -   **Step 2 (Train)**: Call `subprocess.run(["pacemaker", "input.yaml"])`.
        -   Timeout set to 24 hours.
    -   **Step 3 (Validate)**: Call `subprocess.run(["pace_test", ...])` to evaluate on the holdout set.
    -   **Step 4 (Parse)**: Read `metrics.json`. Return a `TrainingResult` object with paths and RMSEs.

## 4. Implementation Approach

1.  **Phase 1: Physics Baseline**
    -   Implement the ZBL screening function coefficients (standard tabulated values).
    -   Write a test comparing the ZBL energy of a dimer at various distances.
    -   Compare against `ase.calculators.lj` (Lennard-Jones) structurally (not numerically).

2.  **Phase 2: Data Export**
    -   This is the most error-prone part. The "Extended XYZ" format is strict.
    -   We will create a small "Golden Dataset" (3 atoms) manually formatted.
    -   We will write a test that exports data and diffs it against the Golden Dataset.

3.  **Phase 3: Templating**
    -   Create `templates/input.yaml.j2`.
    -   Use `jinja2` to render it. Verify that changing `TrainConfig.cutoff` changes the output file.

4.  **Phase 4: Integration (Mocked)**
    -   Since we cannot compile Pacemaker in the CI environment (it requires C++17 and TensorFlow libraries), we will mock the binary.
    -   **Mock**: A Python script `mock_pacemaker.py` that reads the input YAML, sleeps 1s, and writes a dummy `.yace` file and a `metrics.json` with fake low RMSE.
    -   Use this mock to verify the wrapper logic (paths, error handling).

## 5. Test Strategy

### 5.1. Unit Testing
-   **Physics**:
    -   Test `ZBLCalculator` on a dimer at distance $r=0.1\AA$ (very high energy).
    -   Test at $r=10.0\AA$ (near zero energy).
    -   Check Force = Negative Gradient of Energy ($F \approx -\Delta E / \Delta x$).
-   **Data Prep**:
    -   Input: Atom with $E_{DFT} = -100$, $E_{ZBL} = +50$.
    -   Output File: Should list Energy as $-150$.
    -   Input: Atom with `force_mask` array.
    -   Output File: Should have `force_weights` column.

### 5.2. Integration Testing
-   **Full Mock Loop**:
    -   Load 10 structures from DB.
    -   Run `DatasetBuilder`. Check `train.xyz` exists and is non-empty.
    -   Run `PacemakerWrapper` (using the Mock binary).
    -   Check `potential.yace` exists.
    -   Check that the metadata in DB is updated with the training RMSE.
-   **Real "Tiny" Training (Local Development Only)**:
    -   If the developer has Pacemaker installed, run a real training on a 5-frame Aluminum dataset.
    -   Verify the potential predicts a stable lattice constant for Al.

### 5.3. Robustness Testing
-   **Empty Dataset**: What if the DB query returns 0 structures? The wrapper should raise `InsufficientDataError` before starting the subprocess.
-   **Divergence**: If the Mock returns an RMSE of $10^6$ eV, the wrapper should raise a `TrainingDivergenceError` and not save the potential.
