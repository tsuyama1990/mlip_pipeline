# Cycle 06 Specification: Orchestration & Inference

## 1. Summary

Cycle 06 completes the automated loop. We implement **Module E: Inference** and the overarching **Workflow Orchestration**.

Now that we have a trained potential (from Cycle 05), we can run molecular dynamics (MD) simulations using LAMMPS. This is where the system actually "learns" about the material properties. However, a machine learning potential is only accurate near the data it was trained on. If the simulation explores a new phase (e.g., liquid) that was not in the training set, the potential's predictions become unreliable (garbage).

To solve this, we implement **Active Learning (On-The-Fly Learning)**:
1.  **Inference**: Run MD with the `.yace` potential.
2.  **Uncertainty Quantification (UQ)**: At every step, we compute the "Extrapolation Grade" ($\gamma$). This metric tells us how "far" the current local atomic environment is from the basis set spanned by the training data. It is essentially a distance in descriptor space.
3.  **Interruption**: If $\gamma > \text{Threshold}$ (e.g., 5.0), we pause the simulation. The potential is effectively saying "I don't know what's happening here."
4.  **Extraction**: We cut out the cluster of atoms responsible for the high uncertainty.
5.  **Feedback**: We send this cluster back to Cycle 04 (DFT). Once calculated, we re-train (Cycle 05), and the potential becomes smarter.

This cycle also implements the `WorkflowManager`, the state machine that automates this entire Generate -> Select -> DFT -> Train -> Inference -> Extract loop.

## 2. System Architecture

### File Structure
**bold** files are to be created or modified.

```
mlip_autopipec/
├── inference/
│   ├── **__init__.py**
│   ├── **runner.py**           # LammpsRunner (MD Execution)
│   ├── **uncertainty.py**      # Log Parser & Decision Logic
│   └── **embedding.py**        # Structure Extraction Logic
├── orchestration/
│   ├── **__init__.py**
│   ├── **workflow.py**         # The Main Loop (State Machine)
│   ├── **task_queue.py**       # Dask Wrapper
│   └── **dashboard.py**        # HTML Report
└── app.py                      # Final 'run' command
```

### Data Dictionary

| Model Name | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **InferenceConfig** | temperature | float | MD temperature (K). |
| | pressure | float | MD pressure (Bar). |
| | timestep | float | Time step (fs). |
| | steps | int | Number of MD steps. |
| | uncertainty_threshold | float | Max gamma allowed before stop. |
| **WorkflowState** | generation | int | Current active learning cycle index. |
| | phase | str | "DFT", "TRAIN", "INFERENCE". |
| | pending_tasks | List[str] | IDs of currently running jobs. |
| **ExtractedStructure** | atoms | Atoms | The cluster. |
| | center_atom_index | int | The atom with high uncertainty. |
| | force_mask | List[bool] | True for core, False for buffer. |

### Component Interaction
-   **`WorkflowManager`** holds **`WorkflowState`**.
-   **`WorkflowManager`** polls **`DatabaseManager`**.
-   **`LammpsRunner`** executes `lmp_serial`.
-   **`EmbeddingExtractor`** processes the `dump.lammpstrj`.

## 3. Design Architecture

### Inference (`LammpsRunner`)
-   **Input**: `in.lammps`. Must load `pair_style pace`.
-   **Compute**: Use `compute extrapolation_grade` (custom compute in Pacemaker-LAMMPS plugin).
-   **Output**: `dump.lammpstrj` and `log.lammps`.
-   **Termination**: We can use LAMMPS's `fix halt` command to stop the simulation automatically if `c_gamma > threshold`. This avoids parsing logs in Python while LAMMPS runs, which is more robust.

### Embedding Extraction (`EmbeddingExtractor`)
-   **Concept**: We don't want to run DFT on the whole 1000-atom MD box. We want a small 50-atom cluster.
-   **Algorithm Steps**:
    1.  **Read Dump**: Load the final frame where MD stopped.
    2.  **Identify Center**: Find atom index $i$ where $\gamma_i = \max(\gamma)$.
    3.  **Neighbor Search**: Use `ase.neighborlist` to find all atoms within $R = R_{cut} + R_{buffer}$.
    4.  **Construct Cell**: Create a new `Atoms` object. If non-periodic (cluster), add vacuum. If periodic, adjust lattice vectors.
    5.  **Force Masking**:
        -   Define "Core" atoms: distance to center $< R_{cut}$. Mask = 1.
        -   Define "Buffer" atoms: $R_{cut} <$ distance $< R_{buffer}$. Mask = 0.
        -   Store mask in `atoms.arrays['force_mask']`.
    6.  **Save**: Return `CandidateData` to be added to DB.

### Workflow Manager (`WorkflowManager`)
-   **States**: `IDLE`, `GENERATION`, `SELECTION`, `DFT`, `TRAINING`, `INFERENCE`.
-   **Logic Loop**:
    1.  **Check DFT**: Are there pending DFT calculations? If yes, wait. If completed count > threshold -> Transition to `TRAINING`.
    2.  **Check Training**: Is a new potential available? If yes -> Transition to `INFERENCE`.
    3.  **Check Inference**: Did MD stop due to uncertainty?
        -   Yes: Run `EmbeddingExtractor` -> Add new structure to DB (Status=`pending`) -> Transition to `DFT`.
        -   No: MD finished successfully. Pipeline Converged? Or Increment Generation and restart?
-   **Concurrency**: Uses `TaskQueue` to submit jobs asynchronously. It doesn't block waiting for DFT.

## 4. Implementation Approach

1.  **Lammps Runner**: Implement `run_md`. Ensure it writes `fix halt` command in the input script.
2.  **Extractor**: Implement `extract_cluster`. This uses `ase.neighborlist`. Add a `force_mask` array to the `atoms.info`.
3.  **Workflow**: Implement the `run_loop` method. It should be a `while True` loop with a `sleep(60)`. It queries the DB count of pending items to decide the state. It relies on the `TaskQueue` to fire-and-forget jobs.
4.  **Dashboard**: A simple function that generates `status.html` with plots of RMSE and Candidate Counts over time.

## 5. Test Strategy

### Unit Testing
-   **Extractor**:
    -   Create a 3x3x3 supercell of Al.
    -   Pick center atom. Extract neighbors within 4.0 A.
    -   Verify the resulting Atoms object has correct number of atoms.
    -   Verify `force_mask` is 1 for center, 0 for edge.
-   **Workflow State Logic**:
    -   Mock the DB count.
    -   If `pending_dft = 10`, `workflow.decide_next_step()` should return `DFT_EXECUTION`.
    -   If `pending_dft = 0` and `new_potential = True`, return `INFERENCE`.

### Integration Testing
-   **Full Loop (Grand Test)**:
    -   Start with 0 data.
    -   Run `mlip-auto run` with a timeout of 10 minutes.
    -   Mock the external binaries (return fake success).
    -   Verify that the system moved from Generation -> DFT -> Training -> Inference -> Extraction.
    -   Verify DB has entries with `config_type="active_learning"`.
