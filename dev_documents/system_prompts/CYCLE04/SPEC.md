# Cycle 04 Specification: Trainer (Pacemaker Integration)

## 1. Summary

Cycle 04 implements the **Trainer** module, the component responsible for fitting the Atomic Cluster Expansion (ACE) potential to the generated DFT data. We integrate the **Pacemaker** library, a powerful tool for constructing ACE potentials.

Key features in this cycle include:
1.  **Pacemaker Wrapper**: Automating the complex command-line interface of Pacemaker (`pace_collect`, `pace_activeset`, `pace_train`).
2.  **Delta Learning**: To enforce physical robustness, we implement a "Hybrid Potential" strategy. The system automatically configures Pacemaker to learn only the *residual* energy difference between the DFT ground truth and a physics-based baseline (Lennard-Jones or ZBL). This ensures that even in data-sparse regions (like nuclear fusion distances), the potential retains a physically sensible repulsive core.
3.  **Active Set Optimization**: Instead of training on all data (which scales poorly), we implement **MaxVol** (Maximum Volume) selection. This algorithm identifies the most "informative" structures (those that maximize the determinant of the descriptor matrix) and uses only this subset for the heavy lifting of training, significantly speeding up the process without sacrificing accuracy.

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # Update TrainerConfig (Pacemaker settings)
├── trainer/
│   ├── **__init__.py**
│   ├── **interface.py**      # Enhanced BaseTrainer
│   ├── **pacemaker_wrapper.py** # Main Pacemaker Controller
│   ├── **dataset_manager.py**   # Data collection & Active Set
│   └── **delta_learning.py**    # Baseline (LJ/ZBL) configuration logic
└── tests/
    └── unit/
        └── **test_trainer.py**
```

## 3. Design Architecture

### 3.1 Pacemaker Wrapper (`trainer/pacemaker_wrapper.py`)
This class manages the interaction with the `pace_` executables or python API (if available).
*   **Dataset Management**: Converts `ase.Atoms` (from Oracle) into Pacemaker's `pckl.gzip` format using `pace_collect`.
*   **Active Set Selection**: Calls `pace_activeset` to select representative structures.
*   **Training**: Calls `pace_train` with the configured hyperparameters (ladder scheme, regularization, etc.).

### 3.2 Delta Learning Configuration (`trainer/delta_learning.py`)
The `DeltaLearning` helper determines the reference potential.
*   **Input**: `elements`, `baseline_type` (LJ or ZBL).
*   **Logic**:
    *   If `ZBL`, it generates the specific ZBL parameters for the element pairs (Z1, Z2).
    *   If `LJ`, it uses standard parameters (sigma, epsilon) or user-provided ones.
*   **Output**: A YAML snippet or Pacemaker config section defining the reference potential to be subtracted before fitting.

### 3.3 Active Set Manager (`trainer/dataset_manager.py`)
*   **Goal**: Reduce training set size while maintaining information.
*   **Method**: Wrapper around `pace_activeset`. It ensures that the selected set covers the structural diversity of the new data.

## 4. Implementation Approach

1.  **Enhance Domain Models**: Update `TrainerConfig` to include Pacemaker-specific parameters (cutoff, order, basis size).
2.  **Implement DeltaLearning**: Write logic to generate the `potential: reference_potential` section of the `input.yaml` for Pacemaker.
3.  **Implement DatasetManager**: Use `subprocess` to call `pace_collect` and `pace_activeset`. Implement logic to merge new data with existing datasets.
4.  **Implement PacemakerWrapper**: Orchestrate the training workflow:
    1.  Update Dataset (Collect).
    2.  Select Active Set (ActiveSet).
    3.  Generate Input YAML (Config + Delta Learning).
    4.  Run Training (Train).
    5.  Parse Output (Load `potential.yace`).
5.  **Integration**: Update `Orchestrator` to call the real `PacemakerTrainer` instead of Mock.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Delta Learning Test**: Verify that asking for "ZBL" for "Fe-Pt" generates the correct atomic numbers and parameters in the config string.
*   **Config Generation Test**: Verify that the generated `input.yaml` for Pacemaker contains all necessary sections (cutoff, chemical species, reference potential).

### 5.2 Integration Testing
*   **Pacemaker Execution**: (Requires Pacemaker installed) Run a small training job on a dummy dataset (e.g., 10 structures). Verify that `pace_train` runs to completion and produces a `.yace` file.
