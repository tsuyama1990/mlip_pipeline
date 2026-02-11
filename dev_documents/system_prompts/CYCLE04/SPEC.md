# Cycle 04 Specification: Trainer (Pacemaker) & Active Learning

## 1. Summary

Cycle 04 implements the **Trainer**, the component that interfaces with the `pacemaker` engine to fit the ACE potential. This cycle is critical for "Data Efficiency" as it implements **Active Set Selection** (D-Optimality). Instead of training on every single DFT result, the system uses linear algebra (MaxVol algorithm) to select only the most informative structures, discarding redundant data.

We also implement **Delta Learning** configuration, ensuring that the ACE potential learns the difference between DFT and a physical baseline (Lennard-Jones or ZBL), which is essential for the system's robustness.

## 2. System Architecture

### File Structure

Files in **bold** are new or modified in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── **__init__.py**
│   │   ├── **base.py**          # BaseTrainer class
│   │   ├── **pacemaker.py**     # Wrapper for pace_train/pace_activeset
│   │   ├── **dataset.py**       # Dataset management (pckl.gzip)
│   │   └── **priors.py**        # LJ/ZBL baseline configuration
│   └── mock.py
├── domain_models/
│   └── config.py                # Updated with TrainerConfig
└── core/
    └── orchestrator.py          # Updated to use Trainer
```

### Component Interaction
1.  **Orchestrator** receives new `DFTResult`s from Oracle.
2.  **Trainer** adds them to the accumulated dataset (`dataset.py`).
3.  **Trainer** runs `pace_activeset` to select the "Active Set".
4.  **Trainer** generates `input.yaml` for Pacemaker, including the Baseline (Prior) definition (`priors.py`).
5.  **Trainer** executes `pace_train`.
6.  **Trainer** returns the path to the new `potential.yace` file.

## 3. Design Architecture

### 3.1. Trainer Configuration (`domain_models/config.py`)

```python
class TrainerConfig(BaseComponentConfig):
    binary_path: str = "pace_train"
    activeset_binary_path: str = "pace_activeset"

    # Physics Constraints
    prior_type: Literal["lj", "zbl", "none"] = "zbl"

    # Hyperparameters
    ace_basis_size: int = 500  # Number of B-basis functions
    max_epochs: int = 1000
    ladder_step: List[int] = [100, 10]  # Step for body-order ladder
```

### 3.2. Dataset Management (`components/trainer/dataset.py`)
Pacemaker uses a specific pickle format (`pandas` DataFrame serialized). We need a robust way to convert our `DFTResult` list into this format.

-   **Input**: `List[DFTResult]`
-   **Output**: `train.pckl.gzip`

### 3.3. Active Set Selection (`components/trainer/pacemaker.py`)
Wrapper around `pace_activeset`.

```python
def select_active_set(dataset_path: Path, max_size: int) -> Path:
    cmd = [
        "pace_activeset",
        "--dataset", str(dataset_path),
        "--max_size", str(max_size),
        "--algorithm", "maxvol"
    ]
    subprocess.run(cmd, check=True)
    return Path("activeset.pckl.gzip")
```

## 4. Implementation Approach

1.  **Implement Dataset Manager**: Create `components/trainer/dataset.py` using `pandas` to structure the data exactly as Pacemaker expects (columns: `energy`, `forces`, `stress`, `ase_atoms`).
2.  **Implement Priors**: Logic to generate the YAML block for `pair_style hybrid/overlay` (used later) and the internal potential config.
3.  **Implement Pacemaker Wrapper**:
    -   `train()`: Calls `pace_train`.
    -   `select()`: Calls `pace_activeset`.
4.  **Mock Trainer**:
    -   For CI, we cannot run `pace_train` (it takes too long or binaries might be missing).
    -   `MockTrainer` should just "touch" a file named `potential.yace` and return success.
5.  **Orchestrator Integration**: Add the training step to the loop.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Dataset Conversion**: Create a `DFTResult`, convert it to DataFrame, save it, load it back, and verify values match.
-   **Config Generation**: Verify that `input.yaml` contains correct keys (e.g., `cutoff`, `b_basis`).

### 5.2. Integration Testing
-   **Mock Training**: Run the full Orchestrator loop (Cycle 01+02+03+04) with Mock components. Verify that a dummy `potential.yace` is created in `potentials/` directory.
-   **Real Training (Optional)**: If `pace_train` is installed, run a tiny fit (10 atoms, 1 epoch) to verify binary invocation works.
