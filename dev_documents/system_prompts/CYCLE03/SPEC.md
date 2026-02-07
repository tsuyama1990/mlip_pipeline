# Cycle 03 Specification: Trainer Integration (Pacemaker)

## 1. Summary

Cycle 03 connects the generated data to the machine learning engine. We will integrate **Pacemaker**, the official implementation of the Atomic Cluster Expansion (ACE).

The **Trainer** module will be responsible for two key tasks:
1.  **Data Management**: Converting the `ase.Atoms` objects (labelled by the Oracle in Cycle 02) into the binary format required by Pacemaker (`.pckl.gzip` DataFrame). This includes merging new data with the accumulated dataset.
2.  **Model Training**: Configuring and executing the `pace_train` command. This involves setting up the Basis Set (BS) definition (cutoff radii, polynomial degrees) and the loss function weights (Energy vs Forces).
3.  **Active Set Selection**: Implementing the interface to `pace_activeset`. This is the "brain" of data efficiency. Instead of training on all generated structures, we use D-Optimality criteria to select a sparse subset of structures that maximize the information gain, keeping the training set compact and efficient.

We will also implement the **Delta Learning** strategy. The Trainer will be configured to learn the *difference* between the DFT energy and a reference potential (ZBL/LJ). This ensures that the final model behaves physically (repulsive) at short distances, even if the ACE part predicts otherwise.

## 2. System Architecture

Files to be created/modified are marked in **bold**.

```
PYACEMAKER/
├── src/
│   └── mlip_autopipec/
│       ├── **trainer/**
│       │   ├── **__init__.py**
│       │   ├── **pacemaker_trainer.py**    # Implements BaseTrainer
│       │   ├── **dataset_manager.py**      # Data conversion logic
│       │   └── **active_set.py**           # Wrapper for pace_activeset
│       ├── config/
│       │   └── **trainer_config.py**       # Pacemaker-specific settings
│       └── utils/
│           └── **subprocess_handler.py**   # Safe CLI execution helper
└── tests/
    ├── **unit/**
    │   └── **test_dataset_manager.py**
    └── **integration/**
        └── **test_pacemaker_binding.py**
```

## 3. Design Architecture

### 3.1. Dataset Manager
This component acts as the ETL (Extract, Transform, Load) layer.
*   **Input**: List of `ase.Atoms` (with E, F, S info).
*   **Process**:
    1.  Validate consistency (Are forces present? Is energy real?).
    2.  Assign unique IDs to structures.
    3.  Convert to Pandas DataFrame format expected by Pacemaker.
    4.  Save as compressed pickle (`.pckl.gzip`).

### 3.2. Pacemaker Trainer
This wrapper manages the `pace_train` CLI.
*   **Configuration**: It must generate a `input.yaml` file for Pacemaker.
    ```yaml
    cutoff: 6.0
    b_basis: ...
    loss:
      kappa: 0.4 (Force weight)
    backend:
      evaluator: tensorpot
    ```
*   **Execution**: Runs `subprocess.run(["pace_train", ...])` and captures stdout/stderr for logging.
*   **Delta Learning**: It ensures that the `potential` section in the config defines the reference potential (e.g., `ZBL`).

### 3.3. Active Set Selector
Wraps `pace_activeset`.
*   **Logic**:
    1.  Takes a large pool of candidates.
    2.  Computes the descriptor matrix $\Psi$.
    3.  Uses MaxVol algorithm to find rows that maximize $\det(\Psi^T \Psi)$.
    4.  Returns indices of the "best" structures.

## 4. Implementation Approach

1.  **Dependencies**: Ensure `pacemaker` (and its ecosystem) is available in the environment. *Note: Since we are in a dev environment, we might need to mock the binary if it's not pip-installable easily.*
2.  **Dataset Logic**: Implement `ase_to_pacemaker` conversion. Write a test that round-trips data (ASE -> Pckl -> ASE) to ensure no precision loss.
3.  **Config Generation**: Create a Jinja2 template or Pydantic serialization for the Pacemaker `input.yaml`.
4.  **Trainer Implementation**: Write the code to call `pace_train`.
    *   **Crucial**: Implement a `DryRun` mode where it generates the `input.yaml` but doesn't actually call the heavy binary (for CI testing).
5.  **Integration**: Wire the `Trainer` into the `Orchestrator`.

## 5. Test Strategy

### 5.1. Unit Testing Approach
*   **Data Conversion**: Create an `ase.Atoms` object with specific forces. Convert it to the Pacemaker format. Load it back and assert `np.allclose(original_forces, loaded_forces)`.
*   **Config Generation**: Verify that `TrainerConfig` correctly renders the YAML file. Check that `cutoff` and `elements` are correctly propagated.

### 5.2. Integration Testing Approach
*   **Full Train Cycle (Mock Binary)**: Create a dummy script named `pace_train` that accepts arguments and writes a valid empty `.yace` file. Use this to test the python wrapper logic.
*   **Active Set Selection**: Create a small dataset of 10 identical structures and 1 unique structure. Run the `active_set` selection logic (if feasible to run the real binary) and verify it picks the unique one.
*   **Delta Learning Check**: Inspect the generated `input.yaml` to ensure the `core_repulsion` or `reference_potential` section is correctly populated when the config flag is enabled.
