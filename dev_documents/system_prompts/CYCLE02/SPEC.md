# Cycle 02 Specification: Structure Generation & Database Management

## 1. Summary

Cycle 02 expands the system to include the "Structure Generator" and "Trainer" modules, along with a "Database Manager". The goal is to establish the complete pipeline for a "One-Shot" training run: generating an initial set of structures, calculating their properties using the Oracle (from Cycle 01), saving them to a dataset, and training a preliminary ACE potential. This creates the feedback loop's foundation.

## 2. System Architecture

Files added/modified (reflecting current codebase):

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       ├── generator.py       # Structure generation settings
│       └── training.py        # Pacemaker settings
├── generator/
│   ├── __init__.py
│   └── builder.py             # StructureBuilder class (replaces StructureGenerator)
├── training/
│   ├── __init__.py
│   └── pacemaker.py           # PacemakerWrapper class
├── orchestration/
│   └── database.py            # DatabaseManager class (centralized location)
└── app.py
```

## 3. Design Architecture

### 3.1 Structure Generator

**`StructureBuilder` (in `generator/builder.py`)**
-   **Responsibilities**: create atomic structures for the initial training set using various strategies (SQS, Distortions, Defects).
-   **Methods**:
    -   `build() -> Iterator[Atoms]`: Orchestrates the generation pipeline (Base -> Distortions -> Defects).

### 3.2 Database Manager

**`DatabaseManager` (in `orchestration/database.py`)**
-   **Responsibilities**: Persistence of training data and DFT results.
-   **Design**:
    -   Wraps `ase.db` (SQLite) for storing atomic structures and metadata.
    -   Ensures type safety and data integrity (no NaNs).
    -   Thread-safe context manager.

### 3.3 Trainer (Pacemaker Wrapper)

**`PacemakerWrapper` (in `training/pacemaker.py`)**
-   **Responsibilities**: Interface with the `pacemaker` CLI.
-   **Methods**:
    -   `train(initial_potential: Path | None) -> TrainingResult`: Executes `pacemaker` training command.
    -   `generate_config() -> Path`: Automatically generates `input.yaml` from `TrainingConfig`.

### 3.4 Configuration

**`TrainingConfig` (in `config/schemas/training.py`)**
-   Fields: `cutoff`, `b_basis_size`, `kappa`, `kappa_f`, `max_iter`, `batch_size`.

## 4. Implementation Approach

1.  **Structure Generator**:
    -   Use `StructureBuilder` to generate `ase.Atoms` based on `SystemConfig`.
    -   Leverage `SQSStrategy`, `DistortionStrategy`, and `DefectStrategy`.

2.  **Database Manager**:
    -   Use `DatabaseManager` to store generated structures and subsequent DFT results.
    -   Implement `save_dft_result(atoms, result, metadata)`.

3.  **Pacemaker Integration**:
    -   Use `PacemakerWrapper` to execute training.
    -   Ensure `pacemaker` binary is called safely via `subprocess`.

4.  **Integration (App)**:
    -   Add `mlip-auto run-cycle-02` command.
    -   **Pipeline Logic**:
        1.  **Generation**: Generate structures using `StructureBuilder`.
        2.  **Oracle**: Run DFT (or Mock DFT) using `QERunner`.
        3.  **Storage**: Save results to DB using `DatabaseManager`.
        4.  **Training**: Train potential using `PacemakerWrapper` on the data in DB.

## 5. Test Strategy

### 5.1 Unit Testing
-   **Generator**: `StructureBuilder` tests covering all strategies.
-   **Database**: `DatabaseManager` tests for CRUD operations and validation.
-   **Trainer**: `PacemakerWrapper` tests for config generation and subprocess calls (mocked).

### 5.2 Integration Testing
-   **Training Loop**:
    -   Mock `QERunner` to return dummy energies/forces.
    -   Mock `PacemakerWrapper` execution to simulate successful training.
    -   Verify the full flow: Generation -> DB -> Training -> Success.
