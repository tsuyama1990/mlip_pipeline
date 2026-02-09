# Cycle 04: Trainer (Pacemaker) Specification

## 1. Summary

Cycle 04 focuses on integrating the **Trainer** module, which interfaces with the **Pacemaker** library to train Atomic Cluster Expansion (ACE) potentials. This module is responsible for managing the training lifecycle: converting DFT data into Pacemaker-compatible formats, selecting the most informative structures using D-optimality (Active Set selection), and executing the training process. A key feature is the support for **fine-tuning**, where an existing potential is updated with new data rather than trained from scratch, significantly reducing computational cost during the active learning loop.

## 2. System Architecture

This cycle focuses on the `components/trainer` package and integration with `pacemaker` CLI tools.

### File Structure

The following file structure will be created. **Bold** files are to be implemented in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`components/`**
            *   **`trainer/`**
                *   **`__init__.py`**
                *   **`base_trainer.py`** (Abstract Base Class)
                *   **`pacemaker_trainer.py`** (Main Implementation)
                *   **`dataset_utils.py`** (Data conversion/IO)

## 3. Design Architecture

### 3.1 Components

#### `BaseTrainer`
Defines the standard interface for training engines.
*   **`train(dataset: list[CalculationResult], initial_potential: Path | None) -> PotentialArtifact`**:
    *   Input: A list of labelled structures and an optional path to a pre-trained potential.
    *   Output: A `PotentialArtifact` object pointing to the newly trained `.yace` file and containing training metrics (RMSE).

#### `PacemakerTrainer`
Concrete implementation for Pacemaker.
*   **`__init__(config: PacemakerTrainerConfig)`**: Sets up training parameters (cutoff, order, etc.).
*   **`update_dataset(new_data: list[CalculationResult]) -> Path`**: Appends new DFT results to the persistent dataset file (`data/accumulated.pckl.gzip`).
*   **`select_active_set(structures: list[Structure], n: int) -> list[Structure]`**: Uses `pace_activeset` to select `n` most informative structures from a larger pool based on D-optimality.
*   **`_run_training(dataset_path: Path, initial_potential: Path | None) -> Path`**: Constructs and executes the `pace_train` command. Handles `input.yaml` generation.

#### `dataset_utils.py`
Helper functions for data manipulation.
*   **`results_to_dataframe(results: list[CalculationResult]) -> pandas.DataFrame`**: Converts internal objects to a DataFrame compatible with Pacemaker's pickle format.
*   **`save_dataset(df: pandas.DataFrame, path: Path)`**: Saves the DataFrame to a gzip-compressed pickle file.

### 3.2 Domain Models

*   **`PotentialArtifact`**:
    *   `path: Path` (Path to .yace file)
    *   `metrics: dict` (Training/Validation RMSE)
    *   `version: str` (e.g., "generation_005")

*   **`PacemakerTrainerConfig`**:
    *   `cutoff: float` (e.g., 5.0 Å)
    *   `order: int` (e.g., 2 or 3)
    *   `elements: list[str]` (e.g., ["Mg", "O"])
    *   `batch_size: int`
    *   `max_epochs: int`

## 4. Implementation Approach

1.  **Dataset Utilities**: Implement `dataset_utils.py` to handle the conversion between `CalculationResult` objects and the Pandas DataFrame format expected by Pacemaker. Ensure compatibility with `ase.io.read/write` if needed.
2.  **Configuration**: Update `config.py` with `TrainerConfig` and `PacemakerTrainerConfig`.
3.  **Active Set Selection**: Implement `select_active_set` by calling the `pace_activeset` executable via `subprocess`. This is critical for data efficiency.
4.  **Training Wrapper**: Implement `PacemakerTrainer._run_training`. This involves generating a `input.yaml` file dynamically based on the config and executing `pace_train`.
5.  **Fine-Tuning**: Add logic to handle the `--initial_potential` flag in `pace_train` command generation.
6.  **Factory**: Register `PacemakerTrainer` in `ComponentFactory`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_dataset_utils.py`**:
    *   Create a list of dummy `CalculationResult` objects.
    *   Convert to DataFrame.
    *   Assert that columns like `energy`, `forces`, `virial` exist and have correct shapes.
    *   Save and load to verify round-trip integrity.
*   **`test_pacemaker_config.py`**:
    *   Initialize `PacemakerTrainer` with specific settings.
    *   Call an internal method to generate the `input.yaml` string.
    *   Verify that YAML content matches the config (e.g., `cutoff: 5.0`).

### 5.2 Integration Testing
*   **Mock Training**:
    *   Requires `pace_train` to be installed (or mocked in CI).
    *   Create a tiny dataset (e.g., 2 structures).
    *   Run `trainer.train(dataset)`.
    *   Verify that a `.yace` file is created.
    *   Verify that `PotentialArtifact` is returned with valid paths.
*   **Active Set Selection**:
    *   Create a pool of 10 random structures.
    *   Call `select_active_set(n=2)`.
    *   Verify that exactly 2 structures are returned.
