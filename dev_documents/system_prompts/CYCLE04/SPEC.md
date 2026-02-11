# Cycle 04 Specification: Trainer (Pacemaker Integration)

## 1. Summary

Cycle 04 implements the **Trainer** module, which interfaces with the Pacemaker engine to construct the Atomic Cluster Expansion (ACE) potential. The Trainer is responsible for managing the training dataset, selecting the most informative structures (Active Learning), and executing the fitting process.

A key feature of this cycle is **Active Set Selection** based on D-Optimality (maximization of the information matrix determinant). Instead of training on all accumulated data, the system intelligently selects a subset of structures that maximize the model's predictive power while minimizing redundancy. This dramatically reduces computational cost. Additionally, the Trainer configures **Delta Learning**, where the model learns the difference between DFT results and a physical baseline (Lennard-Jones/ZBL), ensuring robustness.

## 2. System Architecture

We expand the `components/trainer` module and introduce Trainer-specific configurations.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**      # Add TrainerConfig (Pacemaker settings)
│       │   └── **results.py**     # Add TrainingResult (RMSE, paths)
│       └── components/
│           ├── **trainer/**
│           │   ├── **__init__.py**
│           │   ├── **base.py**        # BaseTrainer (Abstract)
│           │   ├── **pacemaker.py**   # Pacemaker CLI Wrapper
│           │   ├── **active_set.py**  # Active Set Selection Logic
│           │   └── **utils.py**       # Dataset format conversion (ASE -> pckl)
│           └── **factory.py**         # Update to include PacemakerTrainer
```

### Key Components
1.  **PacemakerWrapper (`src/mlip_autopipec/components/trainer/pacemaker.py`)**: The main driver. It wraps the command-line tools provided by Pacemaker: `pace_collect`, `pace_activeset`, and `pace_train`. It handles file paths and argument construction.
2.  **ActiveSetSelector (`src/mlip_autopipec/components/trainer/active_set.py`)**: Implements the logic to run `pace_activeset`. It takes a large pool of candidate structures and returns the indices of the selected subset.
3.  **DatasetManager (`src/mlip_autopipec/components/trainer/utils.py`)**: Converts Python `Atoms` objects (from the Oracle) into the serialized `pandas.DataFrame` or `pckl.gzip` format required by Pacemaker. It also handles merging new data into the existing dataset.

## 3. Design Architecture

### 3.1. Domain Models
*   **TrainerConfig**:
    *   `ace_basis_config`: Path to the YAML file defining the basis set (B-basis).
    *   `delta_learning`: Boolean (enable/disable).
    *   `baseline_potential`: Config for LJ/ZBL (epsilon, sigma).
    *   `active_set_size`: Target number of structures (e.g., 500).
    *   `max_epochs`: Training iterations.
*   **TrainingResult**:
    *   `potential_path`: Path to the generated `potential.yace`.
    *   `metrics`: Dictionary of RMSE (Energy, Force) on training/validation sets.

### 3.2. Active Set Selection (D-Optimality)
The workflow is:
1.  **Collect**: Gather all labeled structures (new + old).
2.  **Compute Descriptors**: Calculate the ACE descriptors ($\mathbf{c}_i$) for all structures.
3.  **Select**: Use the MaxVol algorithm (via `pace_activeset`) to find the subset of structures that maximizes $\det(\mathbf{M}^T \mathbf{M})$.
4.  **Filter**: Keep only these structures for the actual fitting step.

### 3.3. Delta Learning Setup
*   **Concept**: $E_{total} = E_{baseline} + E_{ACE}$.
*   **Implementation**: The Trainer must configure Pacemaker to subtract the baseline energy from the DFT energy *before* fitting.
*   **Baseline**: Typically a ZBL potential for short-range repulsion or a simple LJ for long-range van der Waals. The Trainer generates the necessary `potential.yaml` configuration for Pacemaker.

## 4. Implementation Approach

1.  **Dependencies**: Ensure `pacemaker` (and its dependencies `pyace`, `tensorpotential`) are available in the environment. *Note: In CI/CD where Pacemaker might not be installable, we must rely heavily on Mocks.*
2.  **Wrapper**: Implement `PacemakerWrapper`. Use `subprocess.run` to call the CLI tools.
    *   `collect(atoms_list, output_path)`
    *   `select_active_set(dataset_path, n_select)`
    *   `train(dataset_path, potential_output_path)`
3.  **Active Set Logic**: Implement the selection logic. Parse the output of `pace_activeset` (which usually writes a new dataset file).
4.  **Delta Learning**: Implement the logic to generate the `baseline` configuration block in the Pacemaker input file.
5.  **Integration**: Update the `Orchestrator` to call the Trainer after the Oracle step.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Wrapper Command Generation**: Verify that the Python code constructs the correct CLI strings (e.g., `pace_train --dataset data.pckl ...`).
*   **Dataset Conversion**: Create a few ASE atoms and verify they are correctly serialized to the Pacemaker-compatible format (using `pandas` or `pickle`).

### 5.2. Integration Testing (Mock Pacemaker)
*   **Goal**: Verify the trainer's logic without the heavy `pace_train` binary.
*   **Mock Implementation**:
    *   Create a `MockTrainer` that pretends to run training.
    *   It should read the input dataset.
    *   It should "produce" a dummy `.yace` file (just an empty file or a simple text file).
    *   It should return a `TrainingResult` with fake RMSE values.
*   **Orchestrator Integration**: Run the full mock loop (Cycle 01 + 02 + 03 + 04 components). Verify that a "potential file" appears in the `potentials/` directory at the end of the cycle.
