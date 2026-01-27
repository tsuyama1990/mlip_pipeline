# Cycle 02 Specification: Structure Generation & Database Management

## 1. Summary

Cycle 02 expands the system to include the "Structure Generator" and "Trainer" modules, along with a "Database Manager". The goal is to establish the complete pipeline for a "One-Shot" training run: generating an initial set of structures, calculating their properties using the Oracle (from Cycle 01), saving them to a dataset, and training a preliminary ACE potential. This creates the feedback loop's foundation.

## 2. System Architecture

Files to be added/modified (in bold):

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       ├── **generator.py**   # Structure generation settings
│       └── **training.py**    # Pacemaker settings
├── **generator/**
│   ├── **__init__.py**
│   └── **structure.py**       # StructureGenerator class
├── **trainer/**
│   ├── **__init__.py**
│   ├── **pacemaker.py**       # PacemakerWrapper class
│   └── **database.py**        # DatabaseManager class
└── app.py
```

## 3. Design Architecture

### 3.1 Structure Generator

**`StructureGenerator` (in `generator/structure.py`)**
-   **Responsibilities**: create atomic structures for the initial training set.
-   **Methods**:
    -   `generate_initial_set(element: str, count: int) -> List[Atoms]`: Creates randomized dimers, trimers, and bulk unit cells with random perturbations (Rattling) to cover basic phase space.

### 3.2 Database Manager

**`DatabaseManager` (in `trainer/database.py`)**
-   **Responsibilities**: Persistence of training data.
-   **Design**:
    -   Stores data as a compressed Pickle file (`.pckl.gzip`) compatible with Pacemaker, or an ASE database (`.db`).
    -   Ensures no duplicate structures (basic fingerprinting).
    -   Validates data integrity (no NaNs in forces) before saving.

### 3.3 Trainer (Pacemaker Wrapper)

**`PacemakerWrapper` (in `trainer/pacemaker.py`)**
-   **Responsibilities**: Interface with the `pacemaker` Python package or CLI.
-   **Methods**:
    -   `train(dataset_path: Path, config: TrainingConfig) -> Path`: Executes `pace_train`.
    -   It must automatically generate the `input.yaml` required by Pacemaker based on `TrainingConfig`.

### 3.4 Configuration

**`TrainingConfig` (in `schemas/training.py`)**
-   Fields: `cutoff`, `order` (ACE order), `batch_size`, `max_epochs`.

## 4. Implementation Approach

1.  **Structure Generator**:
    -   Implement logic to generate `ase.Atoms`. Use `ase.build.bulk` and `ase.Atoms(positions=...)` for clusters.
    -   Apply random thermal noise (displacement) to create variety.

2.  **Database Manager**:
    -   Implement `save_dataset(atoms_list: List[Atoms], path: Path)`.
    -   Implement `load_dataset(path: Path) -> List[Atoms]`.

3.  **Pacemaker Integration**:
    -   Since Pacemaker is executed via CLI tools (`pace_train`), use `subprocess` or direct library calls if available and stable.
    -   Ensure the `potential.yace` file is correctly located after training.

4.  **Integration (App)**:
    -   Add `mlip-auto run-cycle-02` command.
    -   Logic: Generator -> Oracle (Mock/Real) -> DB -> Trainer.

## 5. Test Strategy

### 5.1 Unit Testing
-   **Generator**: Call `generate_initial_set`. Verify the returned list has the requested number of structures. Verify structures are valid `ase.Atoms`.
-   **Database**: Save a list of atoms. Load it back. Assert `positions` and `forces` are identical.

### 5.2 Integration Testing
-   **Training Loop**:
    1.  Generate 10 dummy structures.
    2.  Assign fake Energy/Forces to them.
    3.  Save to DB.
    4.  Call `PacemakerWrapper.train`.
    5.  Assert that a `*.yace` file is created and is not empty.
