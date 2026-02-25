# Cycle 03 Specification: MACE Fine-tuning & Surrogate Generation

## 1. Summary
Cycle 03 implements the core "Teacher" improvement phase (Step 3) and the high-throughput "Student" data generation (Step 4).

Once the Active Learning loop (Step 2) identifies critical structures and labels them via DFT, we must fine-tune the **MACE** foundation model to specialize on this new data. We will implement the `MaceTrainer` module to handle this transfer learning process.

After fine-tuning, the MACE model becomes a powerful "Surrogate Oracle". We will use it to drive Molecular Dynamics (MD) simulations via the new `DynamicsEngine`. These simulations explore the configuration space much more broadly than DFT ever could, generating thousands of diverse structures that will form the training set for the final ACE potential.

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified.

```text
src/pyacemaker/
├── trainer/
│   ├── **__init__.py**
│   ├── **base.py**           # BaseTrainer Interface
│   └── **mace_trainer.py**   # MACE Fine-tuning Logic
├── modules/
│   └── **dynamics_engine.py**# MD (ASE/LAMMPS) Wrapper
└── orchestrator.py           # Added: Step 3 & 4 Workflow
```

## 3. Design Architecture

### 3.1. MACE Trainer (`trainer/mace_trainer.py`)
-   **`MaceTrainer`**: Implements `BaseTrainer`.
    -   **`train(dataset: Dataset, config: TrainingConfig)`**:
        -   Loads the base MACE model.
        -   Freezes early layers (optional).
        -   Fine-tunes the readout layers on the `dft_dataset`.
        -   Saves the new model as `fine_tuned_mace.model`.
    -   **Mock Mode**: Just copies the base model to the new path and logs "Training complete".

### 3.2. Dynamics Engine (`modules/dynamics_engine.py`)
-   **`DynamicsEngine`**:
    -   **`run_md(structure: StructureData, calculator: Calculator, steps: int, temp: float)`**:
        -   Sets up an NVT/NPT simulation using ASE.
        -   Uses the fine-tuned `MaceSurrogateOracle` as the calculator.
        -   Runs for `steps` timesteps.
        -   **Key Feature**: Detects unphysical events (e.g., atoms flying apart or overlapping) and halts early if necessary.
        -   **Output**: A trajectory file (`trajectory.xyz`) or list of `StructureData`.

## 4. Implementation Approach

1.  **Implement Trainer**: Create `MaceTrainer` in `trainer/mace_trainer.py`. Implement the `train` method using `mace.tools.train`.
2.  **Implement Dynamics**: Create `DynamicsEngine` in `modules/dynamics_engine.py`. Use `ase.md.velocitydistribution` and `ase.md.langevin`.
3.  **Update Orchestrator**:
    -   Add `run_step3_mace_finetune()`: Calls `MaceTrainer`.
    -   Add `run_step4_surrogate_sampling()`: Loads `fine_tuned_mace.model`, instantiates `DynamicsEngine`, runs MD on multiple initial structures, and collects the trajectory frames.
    -   Store the trajectory frames as `surrogate_candidates`.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Trainer Interface**: Verify that `MaceTrainer` accepts a `Dataset` and `Config` and produces a model file (mocked).
-   **Dynamics Engine**: Verify that `DynamicsEngine` can run a short MD (e.g., 10 steps) using a Mock Calculator and return a list of 10 `StructureData` objects.

### 5.2. Integration Testing
-   **Step 3-4 Flow**:
    1.  Initialize Orchestrator with the `dft_dataset` from Cycle 2.
    2.  Run Step 3 (Fine-tune). Verify `fine_tuned_mace.model` exists.
    3.  Run Step 4 (MD Sampling). Verify that `surrogate_candidates` contains > 0 structures.
    4.  Verify that the structures in `surrogate_candidates` have reasonable geometries (no atom overlap).
