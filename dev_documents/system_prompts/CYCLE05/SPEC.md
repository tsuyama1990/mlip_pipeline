# Cycle 05 Specification: Active Learning & Training

## 1. Summary

Cycle 05 implements **Module D: Active Learning & Training**. At this stage, we have high-quality DFT data stored in our database (from Cycle 02). Now we must turn that data into a fast, accurate Machine Learning Potential (MLIP). We utilize **Pacemaker**, a state-of-the-art tool for training ACE (Atomic Cluster Expansion) potentials.

This cycle is not just about running a training script. It involves:
1.  **Dataset Preparation**: Converting ASE-db content into the specific formats required by Pacemaker (`.pckl.gzip` or `.extxyz`). This is non-trivial as it requires splitting data into Train/Test sets based on rigorous criteria (e.g., ensuring all phases are represented in both) and ensuring unit consistency.
2.  **Delta Learning**: Implementing the logic to subtract a physics-based baseline (ZBL potential) from the DFT energies/forces. This ensures the ML model learns the *correction* to physics, rather than trying to learn physics from scratch. This is critical for stability at short distances.
3.  **Active Learning Loop**: The training is part of a loop. The module must be able to version potentials (Gen 1, Gen 2...) and track their performance (RMSE) over time. It must be able to "Warm Start" training if supported, or manage the retraining cadence.

By the end of this cycle, the system will be able to take a populated database and autonomously produce a `.yace` file that is ready for MD simulation.

## 2. System Architecture

New components in `src/training`.

```ascii
mlip_autopipec/
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── **pacemaker.py**    # Wrapper for Pacemaker executable
│   │   ├── **dataset.py**      # Data querying and formatting
│   │   ├── **config_gen.py**   # Jinja2 template handling for input.yaml
│   │   └── **physics.py**      # ZBL / Baseline calculations
│   └── config/
│       └── models.py           # Updated with TrainConfig
├── tests/
│   └── training/
│       ├── **test_pacemaker.py**
│       ├── **test_dataset.py**
│       └── **test_physics.py**
```

### Key Components

1.  **`DatasetBuilder`**: Connects to `DatabaseManager`. Queries for "finished" calculations. Splits them (Train/Val/Test). Computes ZBL baseline if requested. Writes files to disk. It ensures that atoms with `force_mask=0` are handled correctly (setting weights to 0).
2.  **`PacemakerWrapper`**: The interface to the `pacemaker` command-line tool. It manages the working directory, generates the `input.yaml` configuration, runs the training, and parses the output logs to extract metrics (RMSE).
3.  **`TrainConfigGenerator`**: Uses `Jinja2` templates to create complex Pacemaker configuration files dynamically, injecting values like `cutoff`, `loss_weights`, and file paths.

## 3. Design Architecture

### Domain Concepts

**Delta Learning**:
$V_{ML}(R) = V_{DFT}(R) - V_{Base}(R)$
We usually use a **ZBL (Ziegler-Biersack-Littmark)** potential as the base. It captures the screened nuclear repulsion at very short distances ($< 1 \text{\AA}$). Without this, ML models (which are polynomial expansions) often fail catastrophically when atoms collide, predicting attractive forces instead of repulsive ones.

**Active Learning Generations**:
We do not overwrite potentials. We version them.
-   `potential_gen0.yace`: Trained on initial SQS/NMS data.
-   `potential_gen1.yace`: Trained on Gen0 + Active Learning Data.
The database must support tagging structures with `generation_id` so we can construct cumulative datasets.

**Weighted Loss**:
$L = w_E |E - \hat{E}|^2 + w_F |F - \hat{F}|^2 + w_S |S - \hat{S}|^2$
Typically, $w_F \gg w_E$ because forces provide $3N$ data points per structure, whereas energy provides only 1.

### Data Models

```python
class TrainConfig(BaseModel):
    cutoff: float = 6.0 # Angstroms
    loss_weights: Dict[str, float] = {"energy": 1.0, "forces": 100.0, "stress": 10.0}
    test_fraction: float = 0.1
    max_generations: int = 10
    enable_delta_learning: bool = True
    batch_size: int = 100
    ace_basis_size: str = "medium" # small, medium, large

class TrainingResult(BaseModel):
    potential_path: Path
    rmse_energy: float
    rmse_forces: float
    training_time: float
    generation: int
```

## 4. Implementation Approach

1.  **Step 1: Physics Logic (`physics.py`)**:
    -   Implement `ZBLCalculator`. It should take an `Atoms` object and return the total ZBL energy and forces. Use `ase.calculators.calculator.Calculator` interface if possible, or a standalone function.
2.  **Step 2: Dataset Logic (`dataset.py`)**:
    -   Implement `DatasetBuilder.fetch_data()`. Use `ase.db` to select rows.
    -   Implement `DatasetBuilder.compute_baseline()`.
    -   Implement `export_to_gzip()`. Iterate through atoms, update arrays (subtract baseline), write to pickle.
    -   Handle `force_mask`. If present in `atoms.arrays`, use it to set `weights`.
3.  **Step 3: Config Templates (`config_gen.py`)**:
    -   Create `src/training/templates/input.yaml.j2`.
    -   It should expose variables for `cutoff`, `path_to_data`, `ladder_step` (ACE parameters).
    -   Use `jinja2` to render this at runtime.
4.  **Step 4: Pacemaker Runner (`pacemaker.py`)**:
    -   Implement `PacemakerWrapper.train(config, data_path)`.
    -   Render the template. Write `input.yaml`.
    -   Run `pacemaker input.yaml`.
    -   Capture stdout. Parse "Final RMSE".
    -   Locate `output.yace`. Return the path.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
-   **ZBL Calculation**: We will test the ZBL logic in isolation. We will create two atoms at very close distance (0.1A). The ZBL energy should be massive (thousands of eV). We will assert that the calculated baseline matches theoretical ZBL values. We will also test that it goes to zero at large distances.
-   **Data Splitting**: We will verify that `DatasetBuilder` creates strictly disjoint training and test sets. We will check that the split is randomized but reproducible (using a seed). We will also check that it handles empty databases gracefully (raising an error).
-   **Template Rendering**: We will render the Jinja2 template with a specific `TrainConfig`. We will parse the resulting YAML string and assert that the `cutoff` and `weights` match the config. We will verify that file paths in the config are correctly escaped (if needed) for the YAML.
-   **Weight handling**: We will create an atom with a partial force mask. We will verify that the exported dataset correctly sets the weight of masked atoms to 0.

### Integration Testing Approach (Min 300 words)
-   **Mock Training**: Training a real ACE potential takes hours. We will mock `subprocess.run(["pacemaker", ...])`.
    -   The mock will read the `input.yaml` (to verify we generated it).
    -   The mock will write a dummy `output.yace` file and a `log.txt` containing "RMSE: 0.005".
    -   We verify that `PacemakerWrapper` parses this log and returns the correct RMSE and file path.
-   **End-to-End Data Flow**:
    1.  Insert 10 dummy DFT results into the DB (Cycle 01).
    2.  Run `DatasetBuilder.export()`.
    3.  Verify the file `train.pckl.gzip` exists.
    4.  Run `PacemakerWrapper.train()`.
    5.  Verify it returns the path to the potential.
    6.  Verify the result object contains the correct Generation index.
