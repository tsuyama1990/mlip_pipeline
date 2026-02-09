# Cycle 04 Specification: Trainer (Pacemaker)

## 1. Summary
The "Trainer" is the module responsible for fitting the Atomic Cluster Expansion (ACE) potential to the labeled data provided by the Oracle. This cycle integrates the external `pacemaker` suite (a Rust/Python hybrid tool) into the Python-based orchestrator. Key features include automating the training process via CLI wrappers and implementing "Active Set Selection" (D-Optimality) to filter redundant structures, ensuring that the training set remains compact and information-rich.

## 2. System Architecture

Files in **bold** are to be created or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── trainer/
│   │   ├── **__init__.py**
│   │   ├── **base.py**             # BaseTrainer (ABC)
│   │   ├── **pacemaker_wrapper.py**# Main Implementation
│   │   └── **active_set.py**       # Active Set Selector
│   └── ...
└── core/
    └── **orchestrator.py**         # Update to use trainer
```

## 3. Design Architecture

### 3.1 Pacemaker Wrapper (`pacemaker_wrapper.py`)

This class manages the interaction with the `pace_train` and `pace_collect` command-line tools.

**Responsibilities:**
*   **Data Conversion**: Taking `List[Structure]` (ASE atoms) and converting them to the `.pckl.gzip` format required by Pacemaker. This may involve using `pace_collect` or writing a custom serializer that mimics it.
*   **Training Execution**: Constructing the command string for `pace_train`.
    *   Arguments: `dataset_path`, `initial_potential` (for fine-tuning), `max_num_epochs` (e.g., 50 for quick updates), `ladder_step` (for body order).
*   **Output Handling**: Parsing the output logs to track RMSE and locating the final `.yace` file.

### 3.2 Active Set Selector (`active_set.py`)

This module implements the logic to select the most informative structures using D-Optimality.

**Algorithm:**
1.  **Input**: A large pool of candidate structures (e.g., 10,000 frames from MD) + an existing "Active Set" (e.g., 500 structures).
2.  **MaxVol Selection**: Use `pace_activeset` command to select $N$ new structures that maximize the determinant of the descriptor matrix (volume of the parallelpiped spanned by feature vectors).
3.  **Output**: A reduced list of structures to be sent to the Oracle for labeling.

### 3.3 Data Management

The Orchestrator must manage the "Golden Dataset".
*   `data/accumulated.pckl.gzip`: The master dataset containing all labeled structures.
*   `data/candidates/`: Temporary storage for structures awaiting selection/labeling.

## 4. Implementation Approach

1.  **Develop Data Converter**: Write a function to convert `ase.Atoms` to Pacemaker's pickle format. This is crucial as passing data via files is the only robust way to interact with the CLI tools.
2.  **Implement Active Set Wrapper**: Wrap `pace_activeset`. Ensure it handles the case where the initial set is empty (Cold Start).
3.  **Implement Trainer Wrapper**: Wrap `pace_train`. Add logic to resume training from a previous checkpoint (`--initial_potential`).
4.  **Integration**: Update `Orchestrator` to call `Trainer.update_dataset()` and `Trainer.train()`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Command Construction**: Verify that `PacemakerWrapper.train(epochs=100)` generates the correct CLI string: `pace_train ... --max_num_epochs 100`.
*   **Data Serialization**: Create a dummy `ase.Atoms` object, save it, and verify `pace_collect` (or `pandas.read_pickle`) can read it.

### 5.2 Integration Testing
*   **End-to-End Training (Mock)**:
    *   Create a tiny dataset (e.g., 5 pre-calculated structures).
    *   Run `Trainer.train()`.
    *   Verify a `potential.yace` file is created.
    *   Verify the trainer exits with code 0.
*   **Active Set Filtering**:
    *   Input: 100 identical structures.
    *   Run `ActiveSetSelector.select(n=5)`.
    *   Verify it returns exactly 5 structures (even if they are identical, MaxVol will pick *something*, though determinant will be near zero. Ideally test with diverse structures).
