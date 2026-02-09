# Cycle 04 Specification: Trainer (The Learner)

## 1. Summary

In this cycle, we implement the **Trainer** module, which interfaces with the **Pacemaker** library to train Machine Learning Interatomic Potentials (MLIPs), specifically Atomic Cluster Expansion (ACE) potentials.

The Trainer is responsible for:
1.  **Data Management**: Converting `ase.Atoms` objects (from the Oracle) into the format required by Pacemaker (pandas DataFrame / pickle).
2.  **Active Set Selection**: Using D-Optimality (via `pace_activeset`) to select the most informative structures from a large pool of candidates, minimizing redundancy and computational cost.
3.  **Potential Fitting**: Running the `pace_train` command to optimize the basis function coefficients.
4.  **Delta Learning**: Implementing the "Physics-Informed" strategy where the ML model learns the *residual* between the DFT energy and a reference physical baseline (Lennard-Jones or ZBL). This ensures robustness in the repulsive core region.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the specific deliverables for this cycle.

```ascii
src/mlip_pipeline/
├── components/
│   ├── trainers/
│   │   ├── **__init__.py**
│   │   ├── **pacemaker.py**    # Pacemaker Wrapper
│   │   ├── **delta.py**        # Baseline Potential Logic (LJ/ZBL)
│   │   └── **utils.py**        # Data conversion (ASE -> Pacemaker)
│   └── base.py                 # (Modified) Enhance BaseTrainer interface
└── domain_models/
    └── **potentials.py**       # (New) Potential Artifact Schema
```

## 3. Design Architecture

### 3.1. Trainer Interface
The `BaseTrainer` in `src/mlip_pipeline/components/base.py` will be refined.

*   `train(self, dataset: List[Structure], initial_potential: Optional[Potential] = None) -> Potential`
    *   Input: A list of labeled structures. Optional path to a previous potential (for fine-tuning).
    *   Output: A `Potential` object containing the path to the `.yace` file and metadata.

### 3.2. Pacemaker Implementation
Located in `src/mlip_pipeline/components/trainers/pacemaker.py`.

*   **Config**: `PacemakerTrainerConfig`.
    *   `cutoff`: Float (e.g., 5.0 Å).
    *   `basis_size`: Dict (e.g., `{'max_deg': 6, 'max_body': 3}`).
    *   `physics_baseline`: Optional[str] (e.g., "ZBL", "LJ").
*   **Logic**:
    1.  **Data Preparation**: Convert `List[Structure]` to `dataset.pckl.gzip`.
    2.  **Active Set Selection**: Run `pace_activeset` to select representative structures (if dataset is large).
    3.  **Fitting**: Construct the `input.yaml` for Pacemaker. Run `pace_train`.
    4.  **Artifact Handling**: The resulting `potential.yace` file is moved to the `potentials/` directory.

### 3.3. Delta Learning (Physics Baseline)
Located in `src/mlip_pipeline/components/trainers/delta.py`.
*   **Concept**: $E_{ML} = E_{DFT} - E_{Baseline}$.
*   **Implementation**:
    *   Before training, calculate $E_{Baseline}$ (using ASE's LJ or ZBL calculator) for every structure.
    *   Subtract this from the reference DFT energy.
    *   Train Pacemaker on the residual.
    *   *Crucially*, the final `.yace` file will only contain the residual. The MD engine must know to add the baseline back during simulation (Hybrid Pair Style).

## 4. Implementation Approach

1.  **Data Utility**: Implement `structures_to_pacemaker_dataframe(structures)` in `utils.py`. This usually involves creating a pandas DataFrame with columns for energy, forces, and virial, and saving it as a pickled gzip.
2.  **Baseline Calculator**: Implement `subtract_baseline(structures, baseline_type="ZBL")`. This modifies the `info['energy']` and `arrays['forces']` of the structures in-place (or copies them).
3.  **Pacemaker Wrapper**: Implement `PacemakerTrainer`. Use `subprocess.run` to call `pace_train` CLI.
    *   *Note*: Pacemaker is primarily a CLI tool. We will generate the YAML config file dynamically.
4.  **Potential Object**: Define the `Potential` class in `domain_models/potentials.py` to store the path and the type of baseline used.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Data Conversion**: Create a list of dummy atoms. Convert to Pacemaker format. Load it back and assert data integrity.
*   **Baseline Subtraction**: Create a structure. Calculate LJ energy. Subtract it. Assert the new "target energy" is lower (or different).

### 5.2. Integration Testing
*   **Mock Training**: Since `pace_train` might be slow or unavailable in strict CI, creating a `MockTrainer` that just writes a dummy `.yace` file is acceptable for pipeline testing.
*   **Real Training (Local)**: If Pacemaker is installed, run a tiny training job (2 structures). Assert `potential.yace` is created.

### 5.3. Requirements
*   Pacemaker must be installed in the environment for the real trainer to work.
