# Cycle 04 Specification: Surrogate Labeling & Pacemaker Base Training

## 1. Summary
Cycle 04 focuses on leveraging the fine-tuned MACE model to generate a massive "Student" dataset (Step 5) and training the initial **ACE** (Atomic Cluster Expansion) potential (Step 6).

The MACE model acts as a "Teacher", labeling thousands of MD structures generated in the previous cycle with high-accuracy energy and force predictions. This effectively transfers the knowledge from MACE to a format usable by the much faster ACE potential.

We will implement the `PacemakerTrainer` module, which wraps the `pacemaker` library/CLI, to train the ACE potential on this surrogate dataset. This potential will be fast but may still contain systematic errors relative to DFT, which we will address in the final cycle (Delta Learning).

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified.

```text
src/pyacemaker/
├── trainer/
│   ├── **pacemaker.py**      # Pacemaker Wrapper Implementation
│   └── **wrapper.py**        # CLI Helper (if needed)
├── oracle/
│   └── mace_oracle.py        # Updated: predict_batch
└── orchestrator.py           # Added: Step 5 & 6 Workflow
```

## 3. Design Architecture

### 3.1. Surrogate Labeling (`oracle/mace_oracle.py`)
-   **`MaceSurrogateOracle.predict_batch(structures: List[StructureData])`**:
    -   Efficiently processes a list of structures.
    -   Uses the MACE model (GPU-accelerated if available) to compute energy, forces, and stress.
    -   **Output**: Updates the `energy`, `forces`, and `stress` fields of the input `StructureData` objects in-place or returns new ones.
    -   **Dataset Creation**: The labeled structures are saved as `surrogate_dataset.pckl.gzip` (standard Pacemaker format).

### 3.2. Pacemaker Trainer (`trainer/pacemaker.py`)
-   **`PacemakerTrainer`**: Implements `BaseTrainer`.
    -   **`train(dataset_path: Path, config: TrainingConfig)`**:
        -   Generates `input.yaml` for Pacemaker, specifying the ACE basis set (B-basis), cutoff, and regularization.
        -   Calls `pacemaker` (subprocess or library) to train the potential.
        -   **Output**: A `potential.yace` file.
    -   **Mock Mode**: Just creates a dummy file named `potential.yace` and sleeps for a few seconds.

## 4. Implementation Approach

1.  **Implement Batch Prediction**: Enhance `MaceSurrogateOracle` in `oracle/mace_oracle.py` to support `predict_batch`.
2.  **Implement Pacemaker Trainer**: Create `PacemakerTrainer` in `trainer/pacemaker.py`. Handle `input.yaml` generation.
3.  **Update Orchestrator**:
    -   Add `run_step5_surrogate_labeling()`: Calls `MaceSurrogateOracle.predict_batch` on `surrogate_candidates` and saves the result to `surrogate_dataset.pckl`.
    -   Add `run_step6_pacemaker_base_training()`: Calls `PacemakerTrainer.train` using `surrogate_dataset.pckl`.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Batch Labeling**: Verify that `predict_batch` returns the same number of structures as input, and that all have `energy` and `forces` populated.
-   **Pacemaker Input Generation**: Verify that `PacemakerTrainer` creates a valid `input.yaml` file (e.g., correct element symbols, cutoff radius).

### 5.2. Integration Testing
-   **Step 5-6 Flow**:
    1.  Initialize Orchestrator with `surrogate_candidates` from Cycle 3.
    2.  Run Step 5 (Labeling). Verify `surrogate_dataset.pckl` exists and is readable.
    3.  Run Step 6 (Training). Verify `potential.yace` exists.
    4.  Verify that the "potential" (even if mocked) can be loaded (file is not empty).
